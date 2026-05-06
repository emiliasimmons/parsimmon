import numpy as np
import pytest

from parsimmon.study import Study


def test_single_group():
    st = Study({"a": {"x": 1}})
    st.branch("G", {"b": {"y": 2}})
    results = list(st)
    assert len(results) == 1
    d = results[0].pars
    assert d["a"]["x"] == 1
    assert d["b"]["y"] == 2


def test_range_expansion():
    st = Study({"a": {"x": Study.arange(3)}})
    st.branch("G", {})
    results = list(st)
    assert len(results) == 3
    assert [d.pars["a"]["x"] for d in results] == [0, 1, 2]


def test_multi_range_cartesian():
    st = Study({"a": {"x": Study.arange(3)}, "b": {"y": Study.each([10, 20])}})
    st.branch("G", {})
    results = list(st)
    assert len(results) == 6
    pairs = {(d.pars["a"]["x"], d.pars["b"]["y"]) for d in results}
    assert pairs == {(x, y) for x in range(3) for y in [10, 20]}


def test_update_group():
    st = Study()
    st.branch("G", {"a": 1, "b": 2})
    st.branch("G", {"b": 99, "c": 3})
    d = next(iter(st)).pars
    assert d["a"] == 1
    assert d["b"] == 99
    assert d["c"] == 3


def test_update_defaults():
    st = Study({"a": 1})
    st.branch("G1", {"b": 10})
    st.branch("G2", {"b": 20})
    st.defaults({"a": 99})
    results = list(st)
    assert all(d.pars["a"] == 99 for d in results)
    assert results[0].pars["b"] == 10
    assert results[1].pars["b"] == 20


def test_clear():
    st = Study()
    st.branch("G", {"a": {"x": 1, "y": 2}, "b": 3})
    st.clear("G", ["a.x"])
    d = next(iter(st)).pars
    assert "x" not in d.get("a", {})
    assert d["a"]["y"] == 2
    assert d["b"] == 3

    st2 = Study({"a": {"x": 1, "y": 2}, "b": 3})
    st2.branch("G", {})
    st2.clear(["a.x", "b"])
    d2 = next(iter(st2)).pars
    assert "x" not in d2.get("a", {})
    assert "b" not in d2
    assert d2["a"]["y"] == 2


def test_len():
    st = Study({"a": Study.arange(3), "b": Study.each([10, 20])})
    st.branch("G1", {})
    st.branch("G2", {"c": 99})
    assert len(st) == 12
    assert len(list(st)) == 12


def test_attr_access():
    st = Study({"a": {"x": 1, "y": 2}})
    st.branch("G", {"a": {"x": 10}, "b": {"z": 3}})
    assert st.a.x == 1
    assert st.a.y == 2
    assert st.G.a.x == 10
    assert st.G.b.z == 3


def test_attr_setattr():
    st = Study()
    st.branch("G", {"a": {"x": 1}})
    st.G.a.y = 99
    assert st.G.a.y == 99
    assert next(iter(st)).pars["a"]["y"] == 99


def test_group_copy():
    st = Study()
    r = Study.arange(3)
    st.branch("G1", {"a": {"b": r}})
    st.branch("G2", st.G1)
    assert list(st.G2.a.b) == [0, 1, 2]
    st.G2.a.c = 99
    assert "c" not in dict(st.G1.a)


def test_collision_detection():
    st = Study()
    for name in ["branch", "branches", "_private"]:
        with pytest.raises(ValueError, match="collides with a Study attribute"):
            st.branch(name, {"x": 1})

    st2 = Study({"a": {"x": 1}})
    with pytest.raises(ValueError, match="collides with default key"):
        st2.branch("a", {"y": 2})

    st3 = Study()
    st3.branch("G", {"x": 1})
    with pytest.raises(ValueError, match="collides with branch name"):
        st3.defaults({"G": {"y": 2}})


def test_short_link():
    st = Study()
    r = Study.arange(3)
    st.branch("G", {"a": {"b": r, "c": st.link("b", lambda x: x * 10)}})
    for d in list(st):
        assert d.pars["a"]["c"] == d.pars["a"]["b"] * 10


def test_range_identity_link():
    st = Study()
    r = Study.arange(3)
    st.branch("G", {"a": {"b": r, "c": r.link(lambda x: x + 100)}})
    for d in list(st):
        assert d.pars["a"]["c"] == d.pars["a"]["b"] + 100


def test_link_chain():
    st = Study()
    r = Study.arange(3)
    st.branch("G", {"a": r, "b": st.link("a", lambda x: x * 2), "c": st.link("b", lambda x: x + 100)})
    for d in list(st):
        assert d.pars["b"] == d.pars["a"] * 2
        assert d.pars["c"] == d.pars["b"] + 100


def test_full_path_link():
    st = Study()
    r = Study.each([0.1, 0.5, 0.9])
    st.branch(
        "G",
        {
            "strains": {"rd": {"init": r}, "ss": {"init": st.link("strains.rd.init", lambda x: round(1 - x, 2))}},
        },
    )
    for d in list(st):
        assert np.isclose(d.pars["strains"]["rd"]["init"] + d.pars["strains"]["ss"]["init"], 1.0, rtol=0.01)
