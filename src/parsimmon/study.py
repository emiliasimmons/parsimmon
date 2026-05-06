"""parsimmon: Study class and Trial NamedTuple."""

import copy
import itertools
import math
from typing import NamedTuple

from ._utils import (
    _deep_update,
    _get_nested,
    _iter_leaves,
    _set_nested,
    _to_nested_objdict,
    objdict,
)
from .ranges import _fmt_params, _ParamLink, _ParamRange
from .ranges import arange as _arange
from .ranges import each as _each
from .ranges import link as _link
from .ranges import linspace as _linspace
from .ranges import logspace as _logspace

_UNRESOLVED = object()


class Trial(NamedTuple):
    """Immutable simulation point yielded by iterating a Study."""

    branch: str
    sim_id: int
    branch_id: int
    pars: objdict


class Study:
    """Base defaults plus named branches of overrides with Cartesian expansion.

    Iteration yields one :class:`Trial` per (branch × range product).
    Attribute access navigates branches then defaults (``st.B2.a.b``).
    """

    def __init__(self, base=None):
        self._base = objdict()
        self._branches = objdict()
        if base is not None:
            _deep_update(self._base, _to_nested_objdict(copy.deepcopy(base)))

    def __getattr__(self, name):
        # prevents infinite recursion during deepcopy when _branches/_base
        # aren't in __dict__ yet
        if name.startswith("_"):
            raise AttributeError(name)
        in_branches = name in self._branches
        in_base = name in self._base
        if in_branches and in_base:
            raise AttributeError(f"'{name}' exists as both branch and default key. Deduplicate")
        if in_branches:
            return self._branches[name]
        if in_base:
            return self._base[name]
        raise AttributeError(f"'{type(self).__name__}' has no branch or default key '{name}'")

    arange = staticmethod(_arange)
    linspace = staticmethod(_linspace)
    logspace = staticmethod(_logspace)
    each = staticmethod(_each)
    iter = staticmethod(_each)  # kept for manager.py CLI override parsing ("iter(...)")
    link = staticmethod(_link)

    def branch(self, name, overrides=None):
        if hasattr(Study, name) or name.startswith("_"):
            raise ValueError(f"Branch name '{name}' collides with a Study attribute")
        if name in self._base:
            raise ValueError(f"Branch name '{name}' collides with default key '{name}'")
        ovr = _to_nested_objdict(copy.deepcopy(overrides)) if overrides is not None else objdict()
        if name in self._branches:
            _deep_update(self._branches[name], ovr)
        else:
            self._branches[name] = ovr

    def _update_base(self, overrides: dict):
        """Merge *overrides* into ``self._base`` in-place."""
        _deep_update(self._base, _to_nested_objdict(copy.deepcopy(overrides)))

    def defaults(self, overrides):
        for key in overrides:
            if key in self._branches:
                raise ValueError(f"Default key '{key}' collides with branch name '{key}'")
        _deep_update(self._base, _to_nested_objdict(copy.deepcopy(overrides)))

    def clear(self, name_or_keys, keys=None):
        if isinstance(name_or_keys, str):
            target = self._branches.get(name_or_keys, objdict())
            key_list = keys or []
        elif isinstance(name_or_keys, (list, tuple)):
            target = self._base
            key_list = name_or_keys
        else:
            raise TypeError(f"Expected str or list, got {type(name_or_keys).__name__}")
        for dotted in key_list:
            path = tuple(dotted.split("."))
            node = target
            for key in path[:-1]:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, dict):
                node.pop(path[-1], None)

    @property
    def branches(self):
        return list(self._branches.keys())

    def _merged(self, name: str) -> objdict:
        return _deep_update(copy.deepcopy(self._base), self._branches[name])

    def __iter__(self):
        # sim_id is a 1-based run-order index across all branches in iteration order,
        # not a stable identifier — it changes if branches are added or reordered.
        sim_id = 1
        for name in self._branches:
            merged = self._merged(name)
            for branch_id, (_, updates) in enumerate(self._expand(merged), start=1):
                yield Trial(branch=name, sim_id=sim_id, branch_id=branch_id, pars=updates)
                sim_id += 1

    def __len__(self):
        return sum(self._count(self._merged(name)) for name in self._branches)

    @staticmethod
    def _expand(merged):
        fixed = {}
        ranges = []
        links = []
        range_paths = {}

        for path, val in _iter_leaves(merged):
            if isinstance(val, _ParamRange):
                ranges.append((path, val))
                range_paths[id(val)] = path
            elif isinstance(val, _ParamLink):
                links.append((path, val))
            else:
                fixed[path] = val

        def _resolve_one(link_obj, updates, link_path):
            source_vals = []
            for sk in link_obj.source:
                if isinstance(sk, _ParamRange):
                    rpath = range_paths.get(id(sk))
                    if rpath is None:
                        raise ValueError(
                            "Linked _ParamRange not found in this branch — "
                            "after st.branch('B', st.Other), link via st.B.x.y, not st.Other.x.y"
                        )
                    dotted = ".".join(rpath)
                    source_vals.append(_get_nested(updates, dotted))

                elif isinstance(sk, _ParamLink):
                    for lp, lk in links:
                        if lk is sk:
                            try:
                                source_vals.append(_get_nested(updates, ".".join(lp)))
                            except KeyError:
                                return _UNRESOLVED
                            break
                    else:
                        raise ValueError("Inner _ParamLink not found in link list")

                elif isinstance(sk, str):
                    if "." not in sk:
                        parent = ".".join(link_path[:-1])
                        dotted = f"{parent}.{sk}" if parent else sk
                    else:
                        dotted = sk
                    try:
                        source_vals.append(_get_nested(updates, dotted))
                    except KeyError:
                        return _UNRESOLVED

                else:
                    raise TypeError(f"Unsupported link source type: {type(sk).__name__}")
            return link_obj.fn(*source_vals)

        def _resolve_links(updates):
            unresolved = list(links)
            for _ in range(len(links) + 1):
                still_unresolved = []
                for path, link_obj in unresolved:
                    val = _resolve_one(link_obj, updates, path)
                    if val is _UNRESOLVED:
                        still_unresolved.append((path, link_obj))
                    else:
                        _set_nested(updates, path, val)
                if not still_unresolved:
                    return
                if len(still_unresolved) == len(unresolved):
                    paths = [".".join(p) for p, _ in still_unresolved]
                    raise ValueError(f"Circular link dependency among: {paths}")
                unresolved = still_unresolved

        def _build_and_resolve(point):
            updates = objdict()
            for path, val in fixed.items():
                _set_nested(updates, path, val)
            for path, val in point.items():
                _set_nested(updates, path, val)
            _resolve_links(updates)
            return updates

        if not ranges:
            yield {}, _build_and_resolve({})
            return

        axes = [[{path: v} for v in pr.values] for path, pr in ranges]

        for combo in itertools.product(*axes):
            point = {k: v for d in combo for k, v in d.items()}
            yield point, _build_and_resolve(point)

    @staticmethod
    def _count(merged):
        sizes = [len(val) for _, val in _iter_leaves(merged) if isinstance(val, _ParamRange)]
        return math.prod(sizes) if sizes else 1

    def print_summary(self):
        branches = list(self._branches)
        for i, name in enumerate(branches):
            merged = self._merged(name)
            lines = [f"{name}: {{"]
            lines.extend(_fmt_params(merged))
            lines.append("}")
            if i > 0:
                print()
            print("\n".join(lines))
        n_branches = len(branches)
        b_label = "branch" if n_branches == 1 else "branches"
        print(f"\nTotal: {len(self)} simulations across {n_branches} {b_label}")
