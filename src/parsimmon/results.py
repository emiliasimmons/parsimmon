"""Lazy, navigable collection of simulation results."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, TypeVar

from ._utils import _get_nested, objdict

T = TypeVar("T")

_UNLOADED = object()
_MISSING = object()


def _get_meta_value(metadata, key):
    first = key.split(".")[0]
    if first in metadata:
        return _get_nested(metadata, key)
    return _get_nested(metadata.pars, key)


class _SimEntry:
    __slots__ = ("_backend", "_cache_key", "_value", "metadata")

    def __init__(self, metadata, value=_MISSING, cache_key=None, backend=None):
        self.metadata = metadata if isinstance(metadata, objdict) else objdict(metadata)
        if value is _MISSING:
            self._value = _UNLOADED if cache_key is not None else None
        else:
            self._value = value
        self._cache_key = cache_key
        self._backend = backend

    def resolve(self) -> Any:
        if self._value is _UNLOADED:
            if self._backend is None or self._cache_key is None:
                raise RuntimeError("No cached result and no in-memory value")
            self._value = self._backend.load(self._cache_key)
        return self._value


def _split_meta(entry):
    """Split entry.metadata into (pars, metadata_without_pars)."""
    pars = entry.metadata.pars
    meta = objdict({k: v for k, v in entry.metadata.items() if k != "pars"})
    return pars, meta


# ---------------------------------------------------------------------------
# Query builder infrastructure
# ---------------------------------------------------------------------------


class _FilterFn:
    """Callable filter with &/| composition and ._expr tag."""

    __slots__ = ("_expr", "_fn")

    def __init__(self, fn, expr):
        self._fn = fn
        self._expr = expr

    def __call__(self, pars, metadata):
        return self._fn(pars, metadata)

    def __and__(self, other):
        return _make_filter_fn(
            lambda p, m, a=self, b=other: a(p, m) and b(p, m),
            f"({self._expr}) & ({other._expr})",
        )

    def __or__(self, other):
        return _make_filter_fn(
            lambda p, m, a=self, b=other: a(p, m) or b(p, m),
            f"({self._expr}) | ({other._expr})",
        )


def _make_filter_fn(fn, expr: str):
    """Wrap a (pars, metadata)->bool function with &/| support and an ._expr tag.

    The returned object is callable.  ``._expr`` is a human-readable string
    describing the filter (e.g. ``"beta == 0.5"``) used by ``Results.__repr__``.
    """
    return _FilterFn(fn, expr)


class _FieldPath:
    """Unbound dotted-path builder.  Comparison operators produce filter fns."""

    __hash__ = None  # __eq__ returns filter fn, not bool

    def __init__(self, path=()):
        self._path = path

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _FieldPath(self._path + (name,))

    def __eq__(self, value):
        dotted = ".".join(self._path)
        return _make_filter_fn(
            lambda pars, metadata, d=dotted, v=value: _get_nested(pars, d) == v,
            f"{dotted} == {value!r}",
        )

    def __ne__(self, value):
        dotted = ".".join(self._path)
        return _make_filter_fn(
            lambda pars, metadata, d=dotted, v=value: _get_nested(pars, d) != v,
            f"{dotted} != {value!r}",
        )

    def __gt__(self, value):
        dotted = ".".join(self._path)
        return _make_filter_fn(
            lambda pars, metadata, d=dotted, v=value: _get_nested(pars, d) > v,
            f"{dotted} > {value!r}",
        )

    def __ge__(self, value):
        dotted = ".".join(self._path)
        return _make_filter_fn(
            lambda pars, metadata, d=dotted, v=value: _get_nested(pars, d) >= v,
            f"{dotted} >= {value!r}",
        )

    def __lt__(self, value):
        dotted = ".".join(self._path)
        return _make_filter_fn(
            lambda pars, metadata, d=dotted, v=value: _get_nested(pars, d) < v,
            f"{dotted} < {value!r}",
        )

    def __le__(self, value):
        dotted = ".".join(self._path)
        return _make_filter_fn(
            lambda pars, metadata, d=dotted, v=value: _get_nested(pars, d) <= v,
            f"{dotted} <= {value!r}",
        )


class _BoundField(_FieldPath):
    """Field selector bound to a Results instance.  Adds .unique() and
    tab-completion via __dir__."""

    def __init__(self, results: Results, path: tuple[str, ...] = ()):
        self._results = results
        self._path = path

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _BoundField(self._results, self._path + (name,))

    def __dir__(self) -> list[str]:
        keys: list[str] = []
        for entry in self._results._entries:
            try:
                d = entry.metadata.pars
                for part in self._path:
                    d = d[part]
                if isinstance(d, dict):
                    for k in d:
                        if k not in keys:
                            keys.append(k)
            except (KeyError, TypeError):
                pass
        return keys

    def unique(self) -> list:
        dotted = ".".join(self._path)
        values: list = []
        for entry in self._results._entries:
            val = _get_nested(entry.metadata.pars, dotted)
            if val not in values:
                values.append(val)
        return sorted(values)


# ---------------------------------------------------------------------------
# Results class
# ---------------------------------------------------------------------------


class Results(Generic[T]):
    """Lazy, navigable collection of simulation results.

    Results load only on iteration or explicit access; grouping,
    filtering, and metadata access never trigger loading.
    """

    def __init__(self, source=None, *, _values=None):
        self._filter_expr = None
        if _values is not None:
            metadata_list, values = _values
            if len(metadata_list) != len(values):
                raise ValueError(
                    f"metadata_list and values must have the same length ({len(metadata_list)} vs {len(values)})"
                )
            self._entries = [_SimEntry(metadata=m, value=v) for v, m in zip(values, metadata_list)]
        elif isinstance(source, (str, Path)):
            from .cache import SimFileCache

            backend = SimFileCache(Path(source))
            index = backend.index()
            self._entries = [_SimEntry(metadata=e, cache_key=e["cache_key"], backend=backend) for e in index]
        elif isinstance(source, list):
            self._entries = list(source)
        elif source is None:
            self._entries = []
        else:
            from .cache import SimCacheBase

            if isinstance(source, SimCacheBase):
                index = source.index()
                self._entries = [_SimEntry(metadata=e, cache_key=e["cache_key"], backend=source) for e in index]
            else:
                raise TypeError(f"Expected str, Path, SimCacheBase, or list, got {type(source).__name__}")

    # -- Iteration & sizing --------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        for entry in self._entries:
            yield entry.resolve()

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return len(self._entries) > 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Results(self._entries[key])
        raise TypeError(f"Results indices must be slices, not {type(key).__name__}")

    # -- Query builder -------------------------------------------------------

    @property
    def P(self):
        return _BoundField(self)

    # -- Navigation ----------------------------------------------------------

    def __getattr__(self, name: str):
        if name.startswith("_") or name in type(self).__dict__:
            raise AttributeError(name)

        parameter_sets = {e.metadata.parameter_set for e in self._entries}
        branches = {e.metadata.branch for e in self._entries}

        # Study wins silently when multiple parameter_sets exist
        if len(parameter_sets) > 1 and name in parameter_sets:
            filtered = [e for e in self._entries if e.metadata.parameter_set == name]
            return Results(filtered)

        if name in branches:
            filtered = [e for e in self._entries if e.metadata.branch == name]
            return Results(filtered)

        available = sorted(parameter_sets | branches)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'; available: {available}")

    @property
    def branches(self):
        buckets: dict[str, list] = {}
        for entry in self._entries:
            key = entry.metadata.branch
            buckets.setdefault(key, []).append(entry)
        return objdict({k: Results(v) for k, v in buckets.items()})

    @property
    def studies(self):
        buckets: dict[str, list] = {}
        for entry in self._entries:
            key = entry.metadata.parameter_set
            buckets.setdefault(key, []).append(entry)
        return objdict({k: Results(v) for k, v in buckets.items()})

    # -- Filtering -----------------------------------------------------------

    def filter(self, predicate) -> Results[T]:
        filtered = [e for e in self._entries if predicate(*_split_meta(e))]
        result = Results(filtered)
        new_expr = getattr(predicate, "_expr", None)
        if self._filter_expr and new_expr:
            result._filter_expr = f"({self._filter_expr}) & ({new_expr})"
        elif self._filter_expr:
            result._filter_expr = self._filter_expr
        elif new_expr:
            result._filter_expr = new_expr
        return result

    # -- Grouping ------------------------------------------------------------

    def groupby(self, *keys) -> Iterator[tuple]:
        def _resolve_key(entry, key):
            if isinstance(key, _FieldPath):
                return _get_nested(entry.metadata.pars, ".".join(key._path))
            return _get_meta_value(entry.metadata, key)

        single = len(keys) == 1
        buckets: dict = {}
        for entry in self._entries:
            if single:
                k = _resolve_key(entry, keys[0])
            else:
                k = tuple(_resolve_key(entry, key) for key in keys)
            buckets.setdefault(k, []).append(entry)

        for k in sorted(buckets.keys()):
            yield k, Results(buckets[k])

    # -- Iteration helpers ---------------------------------------------------

    def first(self) -> tuple[tuple[objdict, objdict], T]:
        if not self._entries:
            raise IndexError("first() called on empty Results")
        entry = self._entries[0]
        pars, meta = _split_meta(entry)
        return (pars, meta), entry.resolve()

    def items(self) -> Iterator[tuple[tuple[objdict, objdict], T]]:
        for entry in self._entries:
            pars, meta = _split_meta(entry)
            yield (pars, meta), entry.resolve()

    def iter_params(self) -> Iterator[tuple[objdict, objdict]]:
        for entry in self._entries:
            yield _split_meta(entry)

    # -- Display -------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._entries)
        branch_names = list(dict.fromkeys(e.metadata.branch for e in self._entries))
        parts = [f"n={n}"]
        if branch_names:
            parts.append(f"branches={branch_names!r}")
        if self._filter_expr:
            parts.append(f"filter='{self._filter_expr}'")
        return f"Results({', '.join(parts)})"

    def _repr_html_(self) -> str:
        n = len(self._entries)
        studies: dict[str, int] = {}
        branches: dict[str, int] = {}
        pars_keys: set[str] = set()
        for entry in self._entries:
            ps = entry.metadata.get("parameter_set", "?")
            br = entry.metadata.get("branch", "?")
            studies[ps] = studies.get(ps, 0) + 1
            branches[br] = branches.get(br, 0) + 1
            if isinstance(entry.metadata.get("pars"), dict):
                pars_keys.update(entry.metadata.pars.keys())

        parts = [f"<div><strong>Results</strong> ({n} entries)"]
        if studies:
            items = ", ".join(f"{k} ({v})" for k, v in studies.items())
            parts.append(f"<br><strong>Studies:</strong> {items}")
        if branches:
            items = ", ".join(f"{k} ({v})" for k, v in branches.items())
            parts.append(f"<br><strong>Branches:</strong> {items}")
        if pars_keys:
            pars_info = []
            for key in sorted(pars_keys):
                values: list = []
                for entry in self._entries:
                    try:
                        val = entry.metadata.pars[key]
                        if val not in values:
                            values.append(val)
                    except (KeyError, TypeError):
                        pass
                try:
                    values = sorted(values)
                except TypeError:
                    pass
                pars_info.append(f"{key}={values}")
            parts.append(f"<br><strong>Parameters:</strong> {', '.join(pars_info)}")
        parts.append("</div>")
        return "".join(parts)
