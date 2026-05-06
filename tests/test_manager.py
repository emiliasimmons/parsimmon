import pytest

from parsimmon.cache import SimFileCache
from parsimmon.manager import Manager
from parsimmon.results import Results
from parsimmon.study import Study

_stub_sim = lambda pars, meta: None


def test_study_bare(pm_simple):
    ps = pm_simple._build("base")
    results = list(ps)
    assert len(results) == 1
    assert results[0].pars["a"] == 1
    assert ps.branches == ["G"]


def test_study_named():
    pm = Manager(_stub_sim)

    @pm.study("Custom")
    def my_fn(ps):
        ps.branch("Custom", {"a": 1})
        return ps

    ps = pm._build("Custom")
    assert next(iter(ps)).pars["a"] == 1


def test_extends():
    pm = Manager(_stub_sim)

    @pm.study
    def parent(ps):
        ps.branch("P", {"a": 1, "b": 2})
        return ps

    @pm.study(extends=parent)
    def child(ps):
        ps.branch("C", {"b": 99, "c": 3})
        return ps

    ps = pm._build("child")
    assert "P" in ps.branches and "C" in ps.branches
    results = list(ps)
    p_result = next(r for r in results if r.branch == "P").pars
    c_result = next(r for r in results if r.branch == "C").pars
    assert p_result["a"] == 1 and p_result["b"] == 2
    assert c_result["b"] == 99 and c_result["c"] == 3


def test_extends_defaults():
    pm = Manager(_stub_sim)

    @pm.study
    def base():
        return {"a": {"x": 1, "y": 2}}

    @pm.study(extends=base)
    def child(ps):
        ps.defaults({"a": {"x": 99}})
        ps.branch("G", {})
        return ps

    d = next(iter(pm._build("child"))).pars
    assert d["a"]["x"] == 99
    assert d["a"]["y"] == 2


def test_zero_arg_builder():
    pm = Manager(_stub_sim)

    @pm.study
    def base():
        return {"a": {"b": 0}}

    @pm.study(extends=base)
    def child(ps):
        ps.branch("G", {"a": {"c": 1}})
        return ps

    d = next(iter(pm._build("child"))).pars
    assert d["a"]["b"] == 0
    assert d["a"]["c"] == 1


def test_extends_chain():
    pm = Manager(_stub_sim)

    @pm.study
    def base():
        return {"a": {"b": 0}}

    @pm.study(extends=base)
    def mid(ps):
        return {"a": {"b": 1}}

    @pm.study(extends=mid)
    def leaf(ps):
        ps.branch("G1", {"a": {"b": 2}})
        ps.branch("G2", {"a": {"b": 3}})
        return ps

    results = list(pm._build("leaf"))
    assert next(r for r in results if r.branch == "G1").pars["a"]["b"] == 2
    assert next(r for r in results if r.branch == "G2").pars["a"]["b"] == 3


def test_extends_unregistered():
    """extends= accepts a plain callable that is NOT decorated with @pm.study."""
    pm = Manager(_stub_sim)

    def base():
        return {"a": {"x": 1, "y": 2}}

    @pm.study(extends=base)
    def child(ps):
        ps.defaults({"a": {"x": 99}})
        ps.branch("G", {"a": {"z": 3}})
        return ps

    d = next(iter(pm._build("child"))).pars
    assert d["a"]["x"] == 99
    assert d["a"]["y"] == 2
    assert d["a"]["z"] == 3
    # base itself should not appear in the registry
    assert "base" not in pm._entries


def test_extends_unregistered_with_study_arg():
    """Unregistered callable returning a Study (with branches) works as extends target."""
    pm = Manager(_stub_sim)

    def base():
        ps = Study()
        ps.branch("P", {"a": 1})
        return ps

    @pm.study(extends=base)
    def child(ps):
        ps.branch("C", {"b": 2})
        return ps

    ps = pm._build("child")
    assert "P" in ps.branches and "C" in ps.branches


def test_return_none_raises():
    pm = Manager(_stub_sim)

    @pm.study
    def bad(ps):
        ps.branch("G", {"a": 1})

    with pytest.raises(TypeError, match="Mapping or Study"):
        pm._build("bad")


def test_cli_overrides():
    pm = Manager(_stub_sim)

    @pm.study
    def grp(ps):
        ps.defaults({"sim": {"dur": 40, "n_agents": 1000}})
        ps.branch("grp", {})
        return ps

    ps = pm._build("grp")
    Manager._apply_cli_overrides(ps, ["sim.dur=20"])
    d = next(iter(ps)).pars
    assert d["sim"]["dur"] == 20
    assert d["sim"]["n_agents"] == 1000


def test_cli_range_overrides():
    pm = Manager(_stub_sim)

    @pm.study
    def grp(ps):
        ps.defaults({"seed": 0, "beta": 0.5})
        ps.branch("grp", {})
        return ps

    cases = [
        ("seed=arange(3)", "seed", [0, 1, 2]),
        ("beta=linspace(0, 1, 3)", "beta", [0.0, 0.5, 1.0]),
        ("beta=logspace(0, 2, 3)", "beta", [1.0, 10.0, 100.0]),
        ("seed=iter([10, 20, 30])", "seed", [10, 20, 30]),
        ("seed=iter(10, 20, 30)", "seed", [10, 20, 30]),
    ]
    for override, key, expected in cases:
        ps = pm._build("grp")
        Manager._apply_cli_overrides(ps, [override])
        assert [r.pars[key] for r in list(ps)] == expected, f"Failed for override: {override}"


def test_analysis_decorator():
    pm = Manager(_stub_sim)

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    @base.analysis
    def analyze(result):
        pass

    assert analyze is base
    assert callable(base.analysis)


@pytest.mark.parametrize("by_ref", [False, True], ids=["by_name", "by_function"])
def test_run(pm_cached, by_ref):
    pm, base = pm_cached
    target = base if by_ref else "base"
    result = pm.run(target)
    assert isinstance(result, Results)
    assert len(result) == 1


def test_run_analysis_only(pm_cached):
    pm, base = pm_cached
    analysis_called = []

    @base.analysis
    def _(result):
        analysis_called.append(result)

    pm.run("base", do_analysis=False)
    result = pm.run(base.analysis)
    assert isinstance(result, Results)
    assert len(analysis_called) == 1


def test_run_analysis_no_cache(tmp_path):
    pm = Manager(lambda pars, meta: None, cache=SimFileCache(tmp_path / "cache"))

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    @base.analysis
    def _(result):
        pass

    with pytest.raises(RuntimeError, match="No cached results"):
        pm.run(base.analysis)


def test_run_force(tmp_path):
    call_count = []

    def sim_fn(pars, meta):
        call_count.append(1)
        return {"v": pars.get("a", 0)}

    pm = Manager(sim_fn, cache=SimFileCache(tmp_path / "cache"))

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    pm.run("base")
    assert len(call_count) == 1
    pm.run("base", force=True)
    assert len(call_count) == 2


def test_run_overrides():
    pm = Manager(lambda pars, meta: {"v": pars.get("a", 0)})

    @pm.study
    def base(ps):
        ps.defaults({"a": 1})
        ps.branch("G", {})
        return ps

    result = pm.run("base", overrides={"a": 99})
    assert [r["v"] for r in result] == [99]


def test_run_do_analysis_false():
    analysis_called = []
    pm = Manager(lambda pars, meta: {"v": 1})

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    @base.analysis
    def _(result):
        analysis_called.append(True)

    pm.run("base", do_analysis=False)
    assert analysis_called == []


def test_results(pm_cached):
    pm, base = pm_cached
    pm.run("base")
    result = pm.results(base)
    assert isinstance(result, Results)
    assert len(result) == 1


def test_results_none(tmp_path):
    pm = Manager(lambda pars, meta: None, cache=SimFileCache(tmp_path / "cache"))

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    with pytest.warns(UserWarning):
        result = pm.results()
    assert not result
    assert len(result) == 0


def _make_list_manager():
    pm = Manager(_stub_sim)

    @pm.study
    def base(ps):
        ps.branch("G1", {"a": Study.arange(3), "b": Study.each([10, 20])})
        ps.branch("G2", {"c": 99})
        return ps

    @pm.study(extends=base)
    def child(ps):
        ps.branch("G3", {"d": 1})
        return ps

    return pm


def test_list_bare(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sim.py"])
    result = _make_list_manager().cli_run(argv=["--list"])
    assert result is None
    assert capsys.readouterr().out.splitlines() == [
        "sim.py (2 parameter sets)",
        "├── base (2 branches, 7 jobs)",
        "│   ├── G1: 6",
        "│   └── G2: 1",
        "└── child (3 branches, 8 jobs)",
        "    ├── G1: 6",
        "    ├── G2: 1",
        "    └── G3: 1",
    ]


def test_list_single(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sim.py"])
    _make_list_manager().cli_run(argv=["--list", "base"])
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["base (2 branches, 7 jobs)", "├── G1: 6", "└── G2: 1"]


def test_list_with_cache(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["sim.py"])
    pm = Manager(lambda pars, meta: {"v": 1}, cache=SimFileCache(tmp_path / "cache"))

    @pm.study
    def base(ps):
        ps.branch("G1", {"a": Study.arange(3), "b": Study.each([10, 20])})
        ps.branch("G2", {"c": 99})
        return ps

    pm.cli_run(argv=["base"])
    capsys.readouterr()

    pm.cli_run(argv=["--list"])
    assert capsys.readouterr().out.splitlines() == [
        "sim.py (1 parameter set)",
        "└── base (2 branches, 7 jobs)",
        "    ├── G1: 6   [cached 6/6]",
        "    └── G2: 1   [cached 1/1]",
    ]


def test_cli_jobs_flag(monkeypatch, pm_cached):
    from unittest.mock import patch

    monkeypatch.setattr("sys.argv", ["sim.py"])
    pm, _ = pm_cached
    captured_jobs = []
    real_execute = pm._execute

    def spy_execute(name, ps, jobs=None, **kwargs):
        captured_jobs.append(jobs)
        return real_execute(name, ps, jobs=jobs, **kwargs)

    with patch.object(pm, "_execute", side_effect=spy_execute):
        pm.cli_run(argv=["-j", "4", "base"])
    assert captured_jobs == [4]


def test_cli_force_flag(tmp_path):
    call_count = []
    pm = Manager(lambda pars, meta: (call_count.append(1), {"v": 1})[1], cache=SimFileCache(tmp_path / "cache"))

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    pm.cli_run(argv=["base"])
    assert len(call_count) == 1
    pm.cli_run(argv=["-f", "base"])
    assert len(call_count) == 2


def test_cli_clean(capsys, tmp_path):
    pm = Manager(lambda pars, meta: {"v": 1}, cache=SimFileCache(tmp_path / "cache"))

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    pm.cli_run(argv=["base"])
    capsys.readouterr()
    pm.cli_run(argv=["--clean", "base"])
    assert "Removed" in capsys.readouterr().out


def test_cli_certify(capsys, tmp_path):
    pm = Manager(lambda pars, meta: {"v": 1}, cache=SimFileCache(tmp_path / "cache"))

    @pm.study
    def base(ps):
        ps.branch("G", {"a": 1})
        return ps

    pm.cli_run(argv=["base"])
    capsys.readouterr()
    pm.cli_run(argv=["--certify", "base"])
    assert "Certified" in capsys.readouterr().out
