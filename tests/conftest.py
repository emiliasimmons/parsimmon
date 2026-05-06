import pytest

from parsimmon.cache import SimFileCache
from parsimmon.manager import Manager

_stub_sim = lambda pars, meta: None


@pytest.fixture
def pm_simple():
    """Manager with a single 'base' set containing one branch."""
    pm = Manager(_stub_sim)

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    return pm


@pytest.fixture
def pm_cached(tmp_path):
    """Manager with a value-returning sim and file cache.

    Returns (pm, base_handle).
    """
    sim_fn = lambda pars, meta: {"v": pars.get("a", 0)}
    cache = SimFileCache(tmp_path / "cache")
    pm = Manager(sim_fn, cache=cache)

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    return pm, base


@pytest.fixture
def cache(tmp_path):
    """SimFileCache in a fresh temp directory."""
    return SimFileCache(tmp_path / "cache")
