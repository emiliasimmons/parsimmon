"""Content-addressed caching for parsimmon simulation runs."""

import abc
import ast
import copy
import hashlib
import importlib.util
import inspect
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np


def _dill_save(filepath, obj):
    with open(filepath, "wb") as f:
        dill.dump(obj, f)


def _dill_load(filepath):
    with open(filepath, "rb") as f:
        return dill.load(f)


def _make_default_serializers():
    return _dill_save, _dill_load


def _canonical_repr_inner(obj, seen):
    """Build deterministic bytes for hashing (recursive helper)."""
    obj_id = id(obj)
    # only track mutable containers that can be circular
    is_container = isinstance(obj, (dict, list, tuple))
    if is_container:
        if obj_id in seen:
            return b"<circular>"
        seen.add(obj_id)

    try:
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: repr(kv[0]))
            parts = [b"dict:{"]
            for k, v in items:
                parts.append(_canonical_repr_inner(k, seen))
                parts.append(b":")
                parts.append(_canonical_repr_inner(v, seen))
                parts.append(b",")
            parts.append(b"}")
            return b"".join(parts)

        if isinstance(obj, (list, tuple)):
            tag = b"list:[" if isinstance(obj, list) else b"tuple:("
            close = b"]" if isinstance(obj, list) else b")"
            parts = [tag]
            for item in obj:
                parts.append(_canonical_repr_inner(item, seen))
                parts.append(b",")
            parts.append(close)
            return b"".join(parts)

        if isinstance(obj, np.ndarray):
            header = f"ndarray:{obj.dtype}:{obj.shape}:".encode()
            return header + obj.tobytes()

        if isinstance(obj, np.bool_):
            obj = bool(obj)
        elif isinstance(obj, np.integer):
            obj = int(obj)
        elif isinstance(obj, np.floating):
            obj = float(obj)

        if isinstance(obj, bool):
            return f"bool:{obj!r}".encode()
        if isinstance(obj, int):
            return f"int:{obj!r}".encode()
        if isinstance(obj, float):
            return f"float:{obj!r}".encode()
        if isinstance(obj, str):
            return f"str:{obj!r}".encode()
        if obj is None:
            return b"None"
        if isinstance(obj, bytes):
            return b"bytes:" + obj

        if callable(obj):
            raise TypeError(
                f"Cannot hash callable {obj!r} for caching. "
                f"Parameterize function selection via a bool, number, or name "
                f"and resolve it to the callable in your own code."
            )

        # fallback: accept any type whose repr is stable across copies
        tname = type(obj).__name__
        r = repr(obj)
        if "0x" in r:
            raise TypeError(f"Cannot hash {tname} for caching: repr contains a memory address")
        try:
            r2 = repr(copy.deepcopy(obj))
        except (TypeError, RecursionError, ValueError, AttributeError, RuntimeError):
            raise TypeError(f"Cannot hash {tname} for caching: deepcopy or repr failed")
        if r != r2:
            raise TypeError(f"Cannot hash {tname} for caching: repr is not stable across copies ({r!r} != {r2!r})")
        return f"obj:{r}".encode()

    finally:
        if is_container:
            seen.discard(obj_id)


def _canonical_repr(obj):
    """Build deterministic bytes for hashing."""
    return _canonical_repr_inner(obj, set())


def hash_params(pars: dict) -> str:
    raw = _canonical_repr(pars)
    return hashlib.sha256(raw).hexdigest()


def compute_cache_key(pars: dict) -> str:
    return hash_params(pars)[:16]


def is_project_local(module_path: Path, project_root: Path) -> bool:
    path_str = str(module_path)
    if "/site-packages/" in path_str or "\\site-packages\\" in path_str:
        return False

    try:
        rel = module_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return False

    venv_markers = {".venv", "venv", "env", ".env", ".tox", ".nox"}
    return not (rel.parts and rel.parts[0] in venv_markers)


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return start.resolve().parent


def _fn_referenced_names(fn: Callable) -> set[str]:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return set()

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _resolve_fn_source_files(fn: Callable) -> list[Path]:
    """Find project-local source files that *fn* depends on at runtime."""
    names = _fn_referenced_names(fn)
    fn_file = inspect.getsourcefile(fn)
    fn_globals = getattr(fn, "__globals__", {})

    seeds: list[Path] = []
    for name in names:
        obj = fn_globals.get(name)
        if obj is None:
            continue
        try:
            src_file = inspect.getsourcefile(obj)
        except (TypeError, OSError):
            continue
        if src_file is None:
            continue

        path = Path(src_file)
        path_str = str(path)
        if "/site-packages/" in path_str or "\\site-packages\\" in path_str:
            continue
        if fn_file and Path(fn_file).resolve() == path.resolve():
            continue

        seeds.append(path)
    return seeds


def hash_function_chain(fn: Callable) -> str:
    """Content hash of the project-local files that *fn* depends on."""
    seed_files = _resolve_fn_source_files(fn)

    if not seed_files:
        # no external project-local deps; fall back to fn source
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            src = ""
        return hashlib.sha256(src.encode()).hexdigest()[:16]

    fn_file = inspect.getsourcefile(fn)
    project_root = find_project_root(Path(fn_file)) if fn_file else Path.cwd()

    collected: dict[str, bytes] = {}
    for seed in seed_files:
        _collect_local_files(seed, project_root, collected, seen=set())

    combined = b"".join(collected[k] for k in sorted(collected))
    return hashlib.sha256(combined).hexdigest()[:16]


def _collect_local_files(file_path: Path, project_root: Path, collected: dict, seen: set) -> None:
    abs_path = str(file_path.resolve())
    if abs_path in seen:
        return
    seen.add(abs_path)

    if not is_project_local(file_path, project_root):
        return

    try:
        content = file_path.read_bytes()
    except OSError:
        return

    collected[abs_path] = content

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        dep_path = _resolve_import_node(node, project_root)
        if dep_path is not None:
            _collect_local_files(dep_path, project_root, collected, seen)


def _resolve_import_node(node: ast.AST, project_root: Path) -> "Path | None":
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module is not None:
        names = [node.module]
    else:
        return None

    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ModuleNotFoundError, ValueError):
            continue
        if spec is None or spec.origin is None:
            continue
        candidate = Path(spec.origin)
        if candidate.suffix == ".py" and is_project_local(candidate, project_root):
            return candidate

    return None


class SimCacheBase(abc.ABC):
    @abc.abstractmethod
    def save(self, cache_key: str, result: Any, metadata: dict) -> None: ...

    @abc.abstractmethod
    def load(self, cache_key: str) -> Any: ...

    @abc.abstractmethod
    def exists(self, cache_key: str) -> bool: ...

    @abc.abstractmethod
    def index(self) -> list[dict]: ...

    @abc.abstractmethod
    def add_index_entry(self, metadata: dict) -> None: ...

    @abc.abstractmethod
    def keys(self) -> list[str]: ...

    @abc.abstractmethod
    def delete(self, cache_key: str) -> None: ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    def certify_entries(self, keys: list[str]) -> int:
        """Mark each entry in *keys* as certified by reloading and re-saving its index info."""
        count = 0
        for key in keys:
            entries = self.index()
            for entry in entries:
                if entry.get("cache_key") == key:
                    entry["certified"] = True
                    self.add_index_entry(entry)
                    count += 1
                    break
        return count

    def clean_entries(self, keys: list[str]) -> int:
        """Delete each entry in *keys* from the cache."""
        count = 0
        for key in keys:
            self.delete(key)
            count += 1
        return count

    def remove_orphans(self) -> int:
        """Delete stored entries whose cache key is not present in the index."""
        indexed_keys = {e.get("cache_key") for e in self.index()}
        orphans = 0
        for key in self.keys():
            if key not in indexed_keys:
                self.delete(key)
                orphans += 1
        return orphans


class SimFileCache(SimCacheBase):
    """File-system cache: ``cache_dir/results/{key}.pkl`` + ``cache_dir/index.cache``.

    Index is held in memory after first read. Writes use atomic rename
    to prevent corruption on crash.
    """

    def __init__(self, directory, save=None, load=None):
        self._dir = Path(directory)
        self._results_dir = self._dir / "results"
        self._index_path = self._dir / "index.cache"
        if save is not None and load is not None:
            self._save, self._load = save, load
        elif save is None and load is None:
            self._save, self._load = _make_default_serializers()
        else:
            raise ValueError("save and load must both be provided or both omitted")
        self._index_cache: list[dict] | None = None

    def _ensure_dirs(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(exist_ok=True)

    def _result_path(self, cache_key: str) -> Path:
        return self._results_dir / f"{cache_key}.pkl"

    def _read_index(self) -> list[dict]:
        if self._index_cache is None:
            if self._index_path.exists():
                self._index_cache = self._load(str(self._index_path))
            else:
                self._index_cache = []
        return self._index_cache

    def _write_index(self, entries: list[dict]) -> None:
        self._ensure_dirs()
        tmp = self._index_path.with_suffix(".cache.tmp")
        self._save(str(tmp), entries)
        os.replace(str(tmp), str(self._index_path))
        self._index_cache = entries

    def exists(self, cache_key: str) -> bool:
        return self._result_path(cache_key).exists()

    def save(self, cache_key: str, result: Any, metadata: dict) -> None:
        self._ensure_dirs()
        path = self._result_path(cache_key)
        if not path.exists():
            self._save(str(path), result)
        self.add_index_entry({**metadata, "cache_key": cache_key})

    def load(self, cache_key: str) -> Any:
        path = self._result_path(cache_key)
        if not path.exists():
            raise KeyError(f"no cached result for key {cache_key!r}")
        return self._load(str(path))

    def add_index_entry(self, metadata: dict) -> None:
        entries = self._read_index()
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata,
        }
        entries.append(entry)
        self._write_index(entries)

    def index(self) -> list[dict]:
        return list(self._read_index())

    def keys(self) -> list[str]:
        if not self._results_dir.exists():
            return []
        return [p.stem for p in self._results_dir.glob("*.pkl")]

    def delete(self, cache_key: str) -> None:
        path = self._result_path(cache_key)
        if path.exists():
            path.unlink()

        entries = self._read_index()
        filtered = [e for e in entries if e.get("cache_key") != cache_key]
        if len(filtered) != len(entries):
            self._write_index(filtered)

    def clear(self) -> None:
        if self._results_dir.exists():
            for p in self._results_dir.glob("*.pkl"):
                p.unlink()
        if self._index_path.exists():
            self._index_path.unlink()
        self._index_cache = None

    def get_fn_hash(self, cache_key: str) -> "str | None":
        for entry in reversed(self._read_index()):
            if entry.get("cache_key") == cache_key:
                return entry.get("fn_hash")
        return None

    def certify_entries(self, new_fn_hash: str, parameter_set: "str | None" = None) -> int:
        entries = self._read_index()
        count = 0
        for entry in entries:
            if parameter_set is not None and entry.get("parameter_set") != parameter_set:
                continue
            if entry.get("fn_hash") != new_fn_hash:
                entry["fn_hash"] = new_fn_hash
                count += 1
        if count:
            self._write_index(entries)
        return count

    def clean_entries(
        self,
        current_fn_hash: "str | None" = None,
        parameter_set: "str | None" = None,
        remove_all: bool = False,
    ) -> int:
        entries = self._read_index()
        to_keep, removed = [], 0
        for entry in entries:
            if parameter_set is not None and entry.get("parameter_set") != parameter_set:
                to_keep.append(entry)
                continue
            if remove_all or (current_fn_hash and entry.get("fn_hash") != current_fn_hash):
                ck = entry.get("cache_key")
                if ck:
                    path = self._result_path(ck)
                    if path.exists():
                        path.unlink()
                removed += 1
            else:
                to_keep.append(entry)
        self._write_index(to_keep)
        return removed

    def remove_orphans(self) -> int:
        if not self._results_dir.exists():
            return 0
        indexed_keys = {e.get("cache_key") for e in self._read_index()}
        orphans = 0
        for p in self._results_dir.glob("*.pkl"):
            if p.stem not in indexed_keys:
                p.unlink()
                orphans += 1
        return orphans
