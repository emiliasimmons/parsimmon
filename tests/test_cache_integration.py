"""Integration tests: cache round-trip with fn_hash invalidation.

Reproduces the structure from hiv-instep-test:
  - sim_fn module imports a helper (like hiv_core) and a plotting module (like plotlib)
  - first run populates cache
  - modifying the plotting module (irrelevant to sim) should NOT trigger a warning
  - modifying the sim helper (relevant to sim) SHOULD invalidate
"""

import importlib.util
import sys
import warnings

import parsimmon as pm
from parsimmon.cache import hash_function_chain


def _run_sim(pars, metadata):
    return {"status": "done", "value": 42}


def _make_manager(sim_fn, cache_dir):
    manager = pm.Manager(sim_fn, cache=pm.SimFileCache(cache_dir))

    @manager.study
    def experiment(ps):
        ps.branch("Control", {"beta": 0.1, "seed": 0})
        ps.branch("Treatment", {"beta": 0.5, "seed": 0})
        return ps

    return manager


def _run_with_manager(cache_dir, sim_fn=_run_sim):
    manager = _make_manager(sim_fn, cache_dir)
    ps = manager._build("experiment")
    return manager._execute("experiment", ps)


def test_first_run_creates_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    result = _run_with_manager(cache_dir)
    assert len(list(result)) == 2
    assert len(list((cache_dir / "results").glob("*.pkl"))) == 2


def test_second_run_uses_cache_no_warnings(tmp_path):
    cache_dir = tmp_path / "cache"
    _run_with_manager(cache_dir)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run_with_manager(cache_dir)

    fn_warnings = [w for w in caught if "function has changed" in str(w.message)]
    assert fn_warnings == []
    assert len(list(result)) == 2


SIM_HELPER_SRC = "def setup(pars):\n    return pars\n"
PLOT_HELPER_SRC = "def make_plot(results):\n    pass\n"
SIM_FN_SRC = 'from sim_helper import setup\nimport plot_helper\n\ndef run(pars, metadata):\n    setup(pars)\n    return {"status": "done", "value": 42}\n'


def _write_project(proj_dir, sim_fn_src=SIM_FN_SRC, sim_helper_src=SIM_HELPER_SRC, plot_helper_src=PLOT_HELPER_SRC):
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / ".git").mkdir(exist_ok=True)
    (proj_dir / "sim_fn.py").write_text(sim_fn_src)
    (proj_dir / "sim_helper.py").write_text(sim_helper_src)
    (proj_dir / "plot_helper.py").write_text(plot_helper_src)


def _load_run_fn(proj_dir):
    """Import the run function from the project's sim_fn.py.

    Registers the module in sys.modules so inspect.getmodule() works,
    matching what happens when ``python basic.py`` runs.
    """
    str_dir = str(proj_dir)
    if str_dir not in sys.path:
        sys.path.insert(0, str_dir)

    for mod_name in ("sim_fn", "sim_helper", "plot_helper"):
        sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location("sim_fn", proj_dir / "sim_fn.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim_fn"] = mod
    spec.loader.exec_module(mod)
    return mod.run


def _run_project(proj_dir, cache_dir):
    return _run_with_manager(cache_dir, sim_fn=_load_run_fn(proj_dir))


def _assert_no_fn_warnings(caught):
    fn_warnings = [w for w in caught if "function has changed" in str(w.message)]
    assert fn_warnings == [], f"Unexpected fn_hash warnings: {[str(w.message) for w in fn_warnings]}"


def test_unchanged_code_no_warnings(tmp_path):
    proj_dir, cache_dir = tmp_path / "project", tmp_path / "cache"
    _write_project(proj_dir)
    _run_project(proj_dir, cache_dir)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_project(proj_dir, cache_dir)
    _assert_no_fn_warnings(caught)


def test_plotting_change_no_warning(tmp_path):
    proj_dir, cache_dir = tmp_path / "project", tmp_path / "cache"
    _write_project(proj_dir)
    _run_project(proj_dir, cache_dir)

    (proj_dir / "plot_helper.py").write_text("def make_plot(results):\n    print('new plot')\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_project(proj_dir, cache_dir)
    _assert_no_fn_warnings(caught)


def test_sim_helper_change_invalidates(tmp_path):
    proj_dir, cache_dir = tmp_path / "project", tmp_path / "cache"
    _write_project(proj_dir)
    _run_project(proj_dir, cache_dir)

    (proj_dir / "sim_helper.py").write_text("def setup(pars):\n    pars['extra'] = True\n    return pars\n")

    result = _run_project(proj_dir, cache_dir)
    assert len(list(result)) == 2


SIM_DEP_SRC = "import copy\n\ndef prepare(pars):\n    return copy.deepcopy(pars)\n"
SIM_DEF_SRC = 'from sim_dep import prepare\n\ndef run_sim(pars, metadata):\n    prepared = prepare(pars)\n    return {"status": "done", "value": 42}\n'
DRIVER_SRC = "from sim_def import run_sim\n\ndef run(pars, metadata):\n    return run_sim(pars, metadata)\n"


def _write_sim_project(proj_dir):
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / ".git").mkdir(exist_ok=True)
    (proj_dir / "sim_dep.py").write_text(SIM_DEP_SRC)
    (proj_dir / "sim_def.py").write_text(SIM_DEF_SRC)
    (proj_dir / "driver.py").write_text(DRIVER_SRC)


def _load_driver(proj_dir):
    str_dir = str(proj_dir)
    if str_dir not in sys.path:
        sys.path.insert(0, str_dir)
    for mod_name in ("driver", "sim_def", "sim_dep"):
        sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location("driver", proj_dir / "driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["driver"] = mod
    spec.loader.exec_module(mod)
    return mod.run


def test_cache_invalidation_on_file_edit(tmp_path):
    proj_dir, cache_dir = tmp_path / "project", tmp_path / "cache"
    _write_sim_project(proj_dir)

    def _run():
        run_fn = _load_driver(proj_dir)
        fn_hash = hash_function_chain(run_fn)
        manager = _make_manager(run_fn, cache_dir)
        ps = manager._build("experiment")
        result = manager._execute("experiment", ps)
        return result, fn_hash

    result, h1 = _run()
    assert len(list(result)) == 2

    # unchanged -> stable hash
    _, h2 = _run()
    assert h2 == h1

    # edit sim_def.py -> hash changes
    with open(proj_dir / "sim_def.py", "a") as f:
        f.write("\n# edited\n")
    _, h3 = _run()
    assert h3 != h1

    # edit sim_dep.py -> hash changes again
    with open(proj_dir / "sim_dep.py", "a") as f:
        f.write("\n# edited\n")
    _, h4 = _run()
    assert h4 != h3

    # stable again
    _, h5 = _run()
    assert h5 == h4
