"""parsimmon: Manager class and CLI helpers."""

import argparse
import ast
import concurrent.futures
import copy
import gc
import inspect
import re
import signal
import sys
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

from ._utils import (
    _get_nested,
    _terminal_menu,
    objdict,
)
from .study import Study


class _Entry(NamedTuple):
    builder: Callable
    parent: str | Callable | None = None


def _init_worker():
    """Ignore SIGINT in worker processes so Ctrl-C is handled only by the parent."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_run_and_save(sim_fn, pars, fn_meta, cache, cache_key):
    """Run one simulation inside a worker process, caching the result before returning.

    Saves the result to disk (if a cache is provided) so the large object is freed
    within the worker process rather than being pickled and sent back over IPC.
    Returns (True, None) when saved, (False, result) when there is no cache.
    """
    result = sim_fn(pars, fn_meta)
    if cache is not None and cache_key is not None:
        cache.save_result(cache_key, result)
        del result
        gc.collect()
        return True, None
    return False, result


class Manager:
    """Binds a simulation function to a registry of named parameter-set
    builders, with decorator API, CLI entry point, optional caching,
    and parallel execution.
    """

    def __init__(self, fn, path=None, cache=None, data_dir=None, plots_dir=None):
        self._fn = fn
        self._entries = objdict()
        self._analyses = {}
        self._extra_args = []
        self._default_name = None
        self._path = path
        self._data_dir_override = Path(data_dir) if data_dir is not None else None
        self._plots_dir_override = Path(plots_dir) if plots_dir is not None else None

        if cache is True:
            from .cache import SimFileCache

            self._cache = SimFileCache(self.data_dir / "cache")
        elif cache:
            self._cache = cache
        else:
            self._cache = None

    def _resolve_dir(self, override, default_name):
        if override is not None:
            return override
        base = Path(default_name)
        return base / self._path if self._path else base

    @property
    def data_dir(self):
        return self._resolve_dir(self._data_dir_override, "data")

    @property
    def plots_dir(self):
        return self._resolve_dir(self._plots_dir_override, "plots")

    def add_argument(self, *args, **kwargs):
        self._extra_args.append((args, kwargs))

    @property
    def _default(self):
        if self._default_name is not None:
            return self._default_name
        return list(self._entries.keys())[-1] if self._entries else None

    def study(self, fn_or_name=None, extends=None, default=False):
        if callable(extends):
            # registered callable -> store by name; unregistered -> store callable directly
            name = getattr(extends, "_pm_name", None)
            parent = name if name is not None and name in self._entries else extends
        else:
            parent = extends

        if callable(fn_or_name):
            return self._register(fn_or_name, fn_or_name.__name__, parent, default)

        reg_name = fn_or_name  # None or str

        def decorator(fn):
            return self._register(fn, reg_name or fn.__name__, parent, default)

        return decorator

    def _register(self, fn, name, parent=None, default=False):
        self._entries[name] = _Entry(fn, parent)
        fn._pm_name = name

        def _analysis_decorator(analysis_fn):
            self._analyses[name] = analysis_fn
            analysis_fn._pm_name = name
            analysis_fn._is_analysis = True
            fn.analysis = analysis_fn
            return fn

        fn.analysis = _analysis_decorator
        if default:
            self._default_name = name
        return fn

    def _fn_to_name(self, fn):
        name = getattr(fn, "_pm_name", None)
        if name is not None and name in self._entries:
            return name
        raise KeyError(f"Function {fn.__name__!r} is not registered")

    def _resolve_name(self, pset):
        if isinstance(pset, str):
            if pset not in self._entries:
                raise KeyError(f"Unknown parameter set {pset!r}")
            return pset
        if callable(pset):
            name = getattr(pset, "_pm_name", None)
            if name is not None and name in self._entries:
                return name
        raise KeyError(f"Cannot resolve parameter set from {pset!r}")

    def _build(self, name):
        entry = self._entries[name]

        if entry.parent:
            if callable(entry.parent):
                ps = self._build_from_callable(entry.parent)
            else:
                ps = copy.deepcopy(self._build(entry.parent))
        else:
            ps = Study()

        sig = inspect.signature(entry.builder)
        needs_ps = any(p.default is p.empty for p in sig.parameters.values())
        result = entry.builder(ps) if needs_ps else entry.builder()

        if isinstance(result, Mapping):
            ps._update_base(result)
        elif isinstance(result, Study):
            ps = result
        else:
            raise TypeError(f"Builder must return Mapping or Study, got {type(result).__name__}")

        return ps

    @staticmethod
    def _build_from_callable(fn):
        """Build a Study from an unregistered callable (used as an extends target)."""
        sig = inspect.signature(fn)
        needs_ps = any(p.default is p.empty for p in sig.parameters.values())
        ps = Study()
        result = fn(ps) if needs_ps else fn()

        if isinstance(result, Mapping):
            ps._update_base(result)
            return ps
        elif isinstance(result, Study):
            return result
        else:
            raise TypeError(f"Builder must return Mapping or Study, got {type(result).__name__}")

    @staticmethod
    def _apply_cli_overrides(ps, overrides):
        for raw in overrides:
            if "=" not in raw:
                raise ValueError(f"Invalid override {raw!r}; expected key=value")
            key, val_str = raw.split("=", 1)
            path = tuple(key.split("."))

            range_val = _parse_range_expr(val_str)
            if range_val is not None:
                val = range_val
            else:
                try:
                    existing = _get_nested(ps._base, key)
                    val = _coerce(val_str, type(existing))
                except (KeyError, ValueError):
                    val = _auto_coerce(val_str)

            nested: dict = {}
            node = nested
            for part in path[:-1]:
                node[part] = {}
                node = node[part]
            node[path[-1]] = val
            ps._update_base(nested)

    def _print_list(self, name=None):
        entries = [name] if name else list(self._entries.keys())

        if self._cache is not None:
            from .cache import compute_cache_key

        if name is None:
            filename = Path(sys.argv[0]).name
            n_sets = len(entries)
            sets_label = "parameter set" if n_sets == 1 else "parameter sets"
            print(f"{filename} ({n_sets} {sets_label})")

        for i, entry_name in enumerate(entries):
            ps_entry = self._build(entry_name)
            branches = ps_entry.branches
            counts = [ps_entry._count(ps_entry._merged(b)) for b in branches]
            total = sum(counts)
            n_branches = len(branches)
            g_label = "branch" if n_branches == 1 else "branches"
            j_label = "job" if total == 1 else "jobs"
            cached_counts = [0] * len(branches)
            if self._cache is not None:
                for bi, b in enumerate(branches):
                    merged = ps_entry._merged(b)
                    for _, pars in ps_entry._expand(merged):
                        if self._cache.exists(compute_cache_key(pars)):
                            cached_counts[bi] += 1

            if name is not None:
                entry_pre, child_pre = "", ""
            else:
                last = i == len(entries) - 1
                entry_pre = "└── " if last else "├── "
                child_pre = "    " if last else "│   "

            print(f"{entry_pre}{entry_name} ({n_branches} {g_label}, {total} {j_label})")
            for j, (b, c) in enumerate(zip(branches, counts)):
                g_con = "└── " if j == len(branches) - 1 else "├── "
                cached_tag = f"   [cached {cached_counts[j]}/{c}]" if cached_counts[j] > 0 else ""
                print(f"{child_pre}{g_con}{b}: {c}{cached_tag}")

    def run(self, pset, jobs=None, force=False, do_analysis=True, overrides=None):
        """Run a parameter set or analysis function.

        Parameters
        ----------
        pset : str, callable, or analysis function
            Name, builder function (from ``@pm.study``), or analysis function
            (from ``@builder.analysis``).
        jobs : int, optional
            Number of parallel workers for simulation execution.
        force : bool
            Re-run all simulations even if cached results exist.
        do_analysis : bool
            Run the registered analysis function after simulation.
        overrides : dict, optional
            Nested dict merged into the study base parameters via deep update.
        """
        if callable(pset) and getattr(pset, "_is_analysis", False):
            name = pset._pm_name
            results = self.results(name)
            if not results:
                raise RuntimeError(f"No cached results for '{name}'; run simulations first")
            print(f"Running analysis for '{name}'...")
            pset(results)
            return results

        name = pset if isinstance(pset, str) else self._fn_to_name(pset)

        ps = self._build(name)
        if overrides:
            ps._update_base(overrides)

        return self._execute(name, ps, jobs, run_analysis=do_analysis, force=force)

    def cli_run(self, jobs=None, argv=None):
        parser = argparse.ArgumentParser()
        default = self._default
        parser.add_argument(
            "name", nargs="?", default=None, choices=list(self._entries.keys()), help="Parameter set to run"
        )
        parser.add_argument(
            "-a",
            "--args",
            action="append",
            nargs="+",
            default=[],
            help="Override key=value pairs",
        )
        parser.add_argument(
            "--print",
            action="store_true",
            dest="print_pars",
            help="Print parameter dicts and exit",
        )
        parser.add_argument(
            "-l",
            "--list",
            action="store_true",
            help="List parameter sets and exit",
        )
        parser.add_argument("-j", "--jobs", type=int, default=None, help="Number of parallel workers")
        parser.add_argument("-f", "--force", action="store_true", help="Force re-run of all simulations")
        parser.add_argument("--clean", action="store_true", help="Remove stale cache entries and exit")
        parser.add_argument("--certify", action="store_true", help="Certify stale cache entries and exit")
        parser.add_argument("-s", "--skip-analysis", action="store_true", help="Skip post-run analysis/plotting")

        for extra_args, extra_kwargs in self._extra_args:
            parser.add_argument(*extra_args, **extra_kwargs)

        args = parser.parse_args(argv)

        if args.list:
            self._print_list(args.name)
            return None

        if args.print_pars:
            self.print_summary(args.name)
            return None

        if args.clean:
            self.clean(args.name)
            return None

        if args.certify:
            self.certify(args.name)
            return None

        resolved_jobs = args.jobs if args.jobs is not None else jobs

        name = args.name or default
        if name is None:
            parser.error("the following arguments are required: name")

        ps = self._build(name)
        cli_overrides = [v for group in args.args for v in group] or None
        if cli_overrides:
            self._apply_cli_overrides(ps, cli_overrides)

        return self._execute(name, ps, resolved_jobs, run_analysis=not args.skip_analysis, force=args.force)

    _RERUN, _CERTIFY, _SKIP, _STOP = range(4)

    _STALENESS_ACTIONS = (_RERUN, _CERTIFY, _SKIP, _STOP)

    def _check_cache_staleness(self, sim_points, fn_hash):
        from .cache import compute_cache_key

        for meta in sim_points:
            ck = compute_cache_key(meta.pars)
            if self._cache.exists(ck):
                stored = self._cache.get_fn_hash(ck)
                if stored is not None and stored != fn_hash:
                    choice = _terminal_menu(
                        "Cache is invalid (simulation script updated):",
                        [
                            "Rerun simulations",
                            "Certify \u2014 keep results and update hash (won't prompt again)",
                            "Skip to analysis",
                            "Stop",
                        ],
                    )
                    return self._STALENESS_ACTIONS[choice]
        return self._RERUN

    def _partition_cache(self, name, sim_points, fn_hash, force, skip_run, certify_run):
        """Split sim_points into cached entries and a run-list.

        Returns ``(entries, to_run)`` where ``entries`` is a list with
        ``_SimEntry`` objects for cache hits (and ``None`` placeholders at the
        positions that must still be executed), and ``to_run`` is a list of
        ``(entry_idx, pars, shared, entry_metadata, cache_key)`` tuples.
        """
        from .cache import compute_cache_key
        from .results import _SimEntry

        entries = []
        to_run = []

        for i, meta in enumerate(sim_points):
            shared = {"parameter_set": name, "branch": meta.branch, "sim_id": meta.sim_id, "branch_id": meta.branch_id}
            entry_metadata = {"pars": copy.deepcopy(meta.pars), "fn_hash": fn_hash, **shared}
            cache_key = None

            if self._cache is not None:
                cache_key = compute_cache_key(meta.pars)
                if force:
                    if self._cache.exists(cache_key):
                        self._cache.delete(cache_key)
                elif self._cache.exists(cache_key):
                    stored_fn_hash = self._cache.get_fn_hash(cache_key)
                    stale = stored_fn_hash is not None and fn_hash != stored_fn_hash
                    if stale and not skip_run:
                        self._cache.delete(cache_key)
                    else:
                        entry_metadata["cache_key"] = cache_key
                        if (certify_run and stale) or not stale:
                            self._cache.add_index_entry(entry_metadata)
                        entries.append(
                            _SimEntry(
                                metadata=entry_metadata,
                                cache_key=cache_key,
                                backend=self._cache,
                            )
                        )
                        continue

            if skip_run:
                continue

            to_run.append((i, meta.pars, shared, entry_metadata, cache_key))
            entries.append(None)

        return entries, to_run

    def _save_entry(self, entries, entry_idx, entry_meta, cache_key, result):
        """Persist a single result and install it in *entries*."""
        from .results import _SimEntry

        if self._cache is not None and cache_key is not None:
            self._cache.save(cache_key, result, entry_meta)
            entry_meta["cache_key"] = cache_key
            entries[entry_idx] = _SimEntry(
                metadata=entry_meta,
                cache_key=cache_key,
                backend=self._cache,
            )
        else:
            entries[entry_idx] = _SimEntry(metadata=entry_meta, value=result)

    def _run_simulations(self, entries, to_run, sim_fn, jobs):
        """Execute pending simulations and cache each result as it completes."""
        if jobs is not None and len(to_run) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as executor:
                future_map = {}
                for run_idx, (_, pars, fn_meta, _, cache_key) in enumerate(to_run):
                    future = executor.submit(_worker_run_and_save, sim_fn, pars, fn_meta, self._cache, cache_key)
                    future_map[future] = run_idx
                try:
                    for future in concurrent.futures.as_completed(future_map):
                        run_idx = future_map[future]
                        entry_idx, _, _, entry_meta, cache_key = to_run[run_idx]
                        _saved, result = future.result()
                        # when saved in worker, result is None but save_result is idempotent
                        self._save_entry(entries, entry_idx, entry_meta, cache_key, result)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
        else:
            for entry_idx, pars, fn_meta, entry_meta, cache_key in to_run:
                result = sim_fn(pars, fn_meta)
                self._save_entry(entries, entry_idx, entry_meta, cache_key, result)
                del result
                gc.collect()

    def _execute(self, name, ps, jobs=None, run_analysis=True, force=False):
        from .results import Results

        if jobs is not None and jobs < 1:
            raise ValueError(f"jobs must be a positive integer, got {jobs}")

        fn = self._fn
        sim_points = list(ps)

        fn_hash = None
        skip_run = False
        certify_run = False
        if self._cache is not None:
            from .cache import hash_function_chain

            fn_hash = hash_function_chain(fn)

            if not force:
                action = self._check_cache_staleness(sim_points, fn_hash)
                if action == self._STOP:
                    raise RuntimeError("Simulation cancelled by user")
                skip_run = action in (self._CERTIFY, self._SKIP)
                certify_run = action == self._CERTIFY

        entries, to_run = self._partition_cache(name, sim_points, fn_hash, force, skip_run, certify_run)

        if to_run:
            n_total = len(sim_points)
            n_cached = n_total - len(to_run)
            parts = []
            if n_cached > 0:
                parts.append(f"cached {n_cached}/{n_total}")
            if jobs is not None:
                parts.append(f"{jobs} jobs")
            suffix = f" ({', '.join(parts)})" if parts else ""
            print(f"Running {len(to_run)}/{n_total} simulations{suffix}...")

            self._run_simulations(entries, to_run, fn, jobs)
        elif skip_run:
            print(f"Loaded {len(entries)} simulations from cache (skipping rerun).")
        else:
            print(f"All {len(sim_points)} simulations loaded from cache.")

        sim_result = Results(entries)

        if run_analysis and name in self._analyses:
            print(f"Running analysis for '{name}'...")
            self._analyses[name](sim_result)

        return sim_result

    def list(self, pset=None):
        name = self._resolve_name(pset) if pset is not None else None
        self._print_list(name)

    def print_summary(self, pset=None):
        if pset is None:
            entry_names = list(self._entries.keys())
        else:
            entry_names = [self._resolve_name(pset)]
        for i, entry_name in enumerate(entry_names):
            if len(entry_names) > 1:
                if i > 0:
                    print()
                print(f"=== {entry_name} ===")
            self._build(entry_name).print_summary()

    def results(self, pset=None):
        from .results import Results, _SimEntry

        if self._cache is None:
            raise RuntimeError("Caching is not enabled; pass cache=True to Manager")

        index = self._cache.index()
        if pset is not None:
            name = self._resolve_name(pset)
            index = [e for e in index if e.get("parameter_set") == name]

        if not index:
            if pset is not None:
                warnings.warn(f"No cached results found for '{name}'")
            else:
                warnings.warn("No cached results found")
            return Results()

        # deduplicate: keep last entry per cache_key
        seen = {}
        for entry in index:
            seen[entry["cache_key"]] = entry
        deduped = list(seen.values())

        entries = [_SimEntry(metadata=e, cache_key=e["cache_key"], backend=self._cache) for e in deduped]
        return Results(entries)

    def certify(self, pset=None):
        if self._cache is None:
            raise RuntimeError("Caching is not enabled; pass cache=True to Manager")
        from .cache import hash_function_chain

        current_hash = hash_function_chain(self._fn)
        name = self._resolve_name(pset) if pset is not None else None
        count = self._cache.certify_entries(current_hash, parameter_set=name)
        print(f"Certified {count} cache entries with current function hash.")

    def clean(self, pset=None, deep=False):
        if self._cache is None:
            raise RuntimeError("Caching is not enabled; pass cache=True to Manager")
        from .cache import hash_function_chain

        name = self._resolve_name(pset) if pset is not None else None
        current_hash = hash_function_chain(self._fn)
        removed = self._cache.clean_entries(
            current_fn_hash=None if deep else current_hash, parameter_set=name, remove_all=deep
        )
        orphans = self._cache.remove_orphans()
        print(f"Removed {removed} cache entries, {orphans} orphan files.")


def _parse_range_expr(val_str):
    m = re.match(r"^(arange|linspace|logspace|iter)\((.+)\)$", val_str.strip())
    if not m:
        return None
    func_name, args_str = m.group(1), m.group(2)

    try:
        args = ast.literal_eval(f"({args_str},)")
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Cannot parse arguments in {val_str!r}") from exc

    func = getattr(Study, func_name)
    if func_name == "iter":
        # iter(iterable) takes one arg; bare multi-args are convenience sugar
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return func(args[0])
        return func(args)
    return func(*args)


def _coerce(val_str, target_type):
    if target_type is bool:
        low = val_str.lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
        raise ValueError(f"Cannot coerce {val_str!r} to bool")
    return target_type(val_str)


def _auto_coerce(val_str):
    for converter in (int, float):
        try:
            return converter(val_str)
        except (ValueError, TypeError):
            pass
    if val_str.lower() in {"true", "false"}:
        return val_str.lower() == "true"
    return val_str
