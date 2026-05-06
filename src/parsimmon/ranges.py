"""parsimmon: parameter range markers and formatting helpers."""

import itertools

import numpy as np

from ._utils import _fmt_scalar, _get_nested, _iter_leaves, objdict


class _ParamRange:
    """Marker for Cartesian expansion; survives deep-merge as a non-dict leaf."""

    def __init__(self, values):
        self.values = list(values)

    def link(self, fn):
        return _ParamLink(self, fn)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"_ParamRange({self.values!r})"


class _ParamLink:
    """Marker for derived parameters, resolved after range expansion."""

    def __init__(self, source, fn):
        self.source = [source] if isinstance(source, (str, _ParamRange, _ParamLink)) else list(source)
        self.fn = fn

    def __repr__(self):
        return f"_ParamLink({self.source!r})"


def _resolve_link_for_display(link, root, range_paths, link_path):
    def _source_values(sk):
        if isinstance(sk, _ParamRange):
            return sk.values
        if isinstance(sk, _ParamLink):
            inner = _resolve_link_for_display(sk, root, range_paths, link_path)
            return inner if isinstance(inner, list) else [inner]
        if "." not in sk:
            parent = ".".join(link_path[:-1])
            dotted = f"{parent}.{sk}" if parent else sk
        else:
            dotted = sk
        sv = _get_nested(root, dotted)
        return sv.values if isinstance(sv, _ParamRange) else [sv]

    sources = [_source_values(sk) for sk in link.source]
    resolved = [link.fn(*combo) for combo in itertools.product(*sources)]
    return resolved if len(resolved) > 1 else resolved[0]


def _range_index(d):
    return {id(val): path for path, val in _iter_leaves(d) if isinstance(val, _ParamRange)}


def _resolve_markers(d, root=None, range_paths=None, prefix=()):
    if root is None:
        root = d
        range_paths = _range_index(d)
    out = objdict()
    for key, val in d.items():
        path = prefix + (key,)
        if isinstance(val, dict):
            out[key] = _resolve_markers(val, root, range_paths, path)
        elif isinstance(val, _ParamRange):
            out[key] = val.values
        elif isinstance(val, _ParamLink):
            out[key] = _resolve_link_for_display(val, root, range_paths, path)
        else:
            out[key] = val
    return out


def _fmt_params(d, root=None, range_paths=None, prefix=(), indent=2):
    if root is None:
        root = d
        range_paths = _range_index(d)
    lines = []
    pad = " " * indent
    for key, val in d.items():
        path = prefix + (key,)
        if isinstance(val, dict):
            lines.append(f"{pad}{key}: {{")
            lines.extend(_fmt_params(val, root, range_paths, path, indent + 2))
            lines.append(f"{pad}}}")
            continue

        if isinstance(val, _ParamRange):
            display = val.values
        elif isinstance(val, _ParamLink):
            display = _resolve_link_for_display(val, root, range_paths, path)
        else:
            display = val

        if isinstance(display, list):
            elems = ", ".join(_fmt_scalar(v) for v in display)
            swept = isinstance(val, (_ParamRange, _ParamLink))
            formatted = f"[[ {elems} ]]" if swept else f"[{elems}]"
        else:
            formatted = _fmt_scalar(display)

        lines.append(f"{pad}{key}: {formatted}")
    return lines


def arange(*args, ndigits=3, **kwargs):
    """Return a _ParamRange from np.arange. Pass ndigits=None to skip rounding."""
    arr = np.arange(*args, **kwargs)
    return _ParamRange(np.round(arr, ndigits) if ndigits is not None else arr)


def linspace(*args, ndigits=3, **kwargs):
    """Return a _ParamRange from np.linspace. Pass ndigits=None to skip rounding."""
    arr = np.linspace(*args, **kwargs)
    return _ParamRange(np.round(arr, ndigits) if ndigits is not None else arr)


def logspace(*args, ndigits=3, **kwargs):
    """Return a _ParamRange from np.logspace. Pass ndigits=None to skip rounding."""
    arr = np.logspace(*args, **kwargs)
    return _ParamRange(np.round(arr, ndigits) if ndigits is not None else arr)


def each(iterable):
    """Wrap an arbitrary iterable as a _ParamRange for Cartesian expansion."""
    return _ParamRange(iterable)


def link(source, fn):
    return _ParamLink(source, fn)
