import pickle

import pytest

from parsimmon._utils import objdict
from parsimmon.cache import SimFileCache
from parsimmon.results import Results


def _make_result(branches_spec):
    values = []
    metadata_list = []
    sim_id = 1
    for branch_name, count, pars in branches_spec:
        for branch_id in range(1, count + 1):
            values.append({"value": sim_id})
            metadata_list.append({
                "parameter_set": "test_set",
                "branch": branch_name,
                "sim_id": sim_id,
                "branch_id": branch_id,
                "pars": objdict(pars),
            })
            sim_id += 1
    return Results(_values=(metadata_list, values))


def _pkl_save(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _pkl_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Retained tests (adapted)
# ---------------------------------------------------------------------------


def test_iter_and_len():
    sr = _make_result([("Baseline", 3, {"beta": 0.5})])
    assert len(sr) == 3
    values = list(sr)
    assert len(values) == 3
    for i, v in enumerate(values, start=1):
        assert v["value"] == i


def test_getitem_slice():
    sr = _make_result([("A", 5, {"x": 1})])
    sliced = sr[1:4]
    assert isinstance(sliced, Results)
    assert len(sliced) == 3
    assert next(iter(sliced)) == {"value": 2}


def test_bool():
    assert not Results([])
    assert _make_result([("A", 1, {})])


def test_attr_branch_access():
    sr = _make_result([("Baseline", 2, {"beta": 0.5}), ("Treatment", 3, {"beta": 0.8})])
    assert len(sr.Baseline) == 2
    assert len(sr.Treatment) == 3
    # Use iter_params instead of .metadata
    assert all(meta["branch"] == "Baseline" for _pars, meta in sr.Baseline.iter_params())


def test_branches_property():
    sr = _make_result([("Baseline", 2, {"beta": 0.5}), ("Treatment", 3, {"beta": 0.8})])
    branches = sr.branches
    assert set(branches) == {"Baseline", "Treatment"}
    assert len(branches["Baseline"]) == 2
    assert len(branches["Treatment"]) == 3


def test_filter_by_predicate():
    sr = _make_result([("Low", 2, {"beta": 0.1}), ("High", 3, {"beta": 0.8})])
    filtered = sr.filter(lambda pars, meta: pars["beta"] > 0.3)
    assert len(filtered) == 3


def test_from_cache(tmp_path):
    cache = SimFileCache(tmp_path, save=_pkl_save, load=_pkl_load)
    for i, key in enumerate(["key001", "key002", "key003"], start=1):
        cache.save(key, {"result": i}, {"branch": "Baseline", "pars": {"beta": 0.5}})

    sr = Results(tmp_path)
    assert len(sr) == 3
    (_pars, _meta), val = sr.first()
    assert val == {"result": 1}

    sr2 = Results(cache)
    assert len(sr2) == 3


def test_repr():
    sr = _make_result([("Baseline", 2, {}), ("Treatment", 3, {})])
    r = repr(sr)
    assert "Results" in r
    assert "n=5" in r
    assert "Baseline" in r
    assert "n=0" in repr(Results([]))


def test_constructor_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        Results(_values=([{"a": 1}, {"b": 2}], [1, 2, 3]))


# ---------------------------------------------------------------------------
# P query builder tests
# ---------------------------------------------------------------------------


def test_P_unique():
    sr = _make_result([("A", 2, {"beta": 0.5}), ("B", 3, {"beta": 0.8})])
    unique_betas = sr.P.beta.unique()
    assert unique_betas == [0.5, 0.8]


def test_P_unique_nested():
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        {
            "parameter_set": "test_set",
            "branch": g,
            "sim_id": i + 1,
            "branch_id": (i % 2) + 1,
            "pars": objdict({"model": {"rate": r}}),
        }
        for i, (g, r) in enumerate([("A", 0.1), ("A", 0.1), ("B", 0.9), ("B", 0.5)])
    ]
    sr = Results(_values=(metadata_list, values))
    unique_rates = sr.P.model.rate.unique()
    assert unique_rates == [0.1, 0.5, 0.9]


def test_P_eq():
    sr = _make_result([("A", 2, {"beta": 0.5}), ("B", 3, {"beta": 0.8})])
    fn = sr.P.beta == 0.5
    # The filter fn is callable
    pars_match = objdict({"beta": 0.5})
    pars_no_match = objdict({"beta": 0.8})
    assert fn(pars_match, {}) is True
    assert fn(pars_no_match, {}) is False


def test_P_gt_lt_ge_le_ne():
    sr = _make_result([("A", 1, {"beta": 0.5})])
    pars = objdict({"beta": 0.5})
    meta = {}

    assert (sr.P.beta > 0.3)(pars, meta) is True
    assert (sr.P.beta > 0.5)(pars, meta) is False
    assert (sr.P.beta < 0.9)(pars, meta) is True
    assert (sr.P.beta < 0.5)(pars, meta) is False
    assert (sr.P.beta >= 0.5)(pars, meta) is True
    assert (sr.P.beta >= 0.6)(pars, meta) is False
    assert (sr.P.beta <= 0.5)(pars, meta) is True
    assert (sr.P.beta <= 0.4)(pars, meta) is False
    assert (sr.P.beta != 0.5)(pars, meta) is False
    assert (sr.P.beta != 0.9)(pars, meta) is True


def test_P_and():
    sr = _make_result([("A", 1, {"beta": 0.5})])
    f1 = sr.P.beta > 0.3
    f2 = sr.P.beta < 0.9
    combined = f1 & f2
    pars = objdict({"beta": 0.5})
    assert combined(pars, {}) is True
    assert (f1 & (sr.P.beta > 0.9))(pars, {}) is False
    # _expr propagates
    assert ">" in combined._expr
    assert "<" in combined._expr


def test_P_or():
    sr = _make_result([("A", 1, {"beta": 0.5})])
    f1 = sr.P.beta < 0.3
    f2 = sr.P.beta > 0.9
    combined = f1 | f2
    pars_low = objdict({"beta": 0.1})
    pars_mid = objdict({"beta": 0.5})
    pars_high = objdict({"beta": 0.95})
    assert combined(pars_low, {}) is True
    assert combined(pars_mid, {}) is False
    assert combined(pars_high, {}) is True
    assert "<" in combined._expr
    assert ">" in combined._expr


def test_P_nested_composition():
    sr = _make_result([("A", 1, {"beta": 0.5})])
    f1 = sr.P.beta > 0.3
    f2 = sr.P.beta < 0.9
    f3 = sr.P.beta < 0.2
    composed = (f1 & f2) | f3
    pars_match = objdict({"beta": 0.5})
    pars_low = objdict({"beta": 0.1})
    pars_high = objdict({"beta": 0.95})
    assert composed(pars_match, {}) is True
    assert composed(pars_low, {}) is True
    assert composed(pars_high, {}) is False


def test_P_dir():
    sr = _make_result([("A", 2, {"beta": 0.5, "alpha": 0.1})])
    keys = dir(sr.P)
    assert "beta" in keys
    assert "alpha" in keys
    # Should not contain non-pars keys
    assert "branch" not in keys


def test_P_expr_tag():
    sr = _make_result([("A", 1, {"beta": 0.5})])
    fn = sr.P.beta == 0.5
    assert hasattr(fn, "_expr")
    assert "beta" in fn._expr
    assert "0.5" in fn._expr


def test_P_is_property():
    sr = _make_result([("A", 1, {"beta": 0.5})])
    # Each access returns a new _BoundField instance
    assert sr.P is not sr.P


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


def test_filter_P_expression():
    sr = _make_result([("A", 2, {"beta": 0.5}), ("B", 3, {"beta": 0.8})])
    filtered = sr.filter(sr.P.beta == 0.5)
    assert len(filtered) == 2
    for pars, _meta in filtered.iter_params():
        assert pars["beta"] == 0.5


def test_filter_P_composed():
    sr = _make_result([("Low", 2, {"beta": 0.1}), ("Mid", 2, {"beta": 0.5}), ("High", 3, {"beta": 0.8})])
    filtered = sr.filter((sr.P.beta > 0.3) & (sr.P.beta < 0.9))
    assert len(filtered) == 5  # Mid(2) + High(3)


def test_filter_lambda_no_expr():
    sr = _make_result([("A", 2, {"beta": 0.5})])
    filtered = sr.filter(lambda pars, meta: pars["beta"] == 0.5)
    assert filtered._filter_expr is None


# ---------------------------------------------------------------------------
# Expression tracking in repr
# ---------------------------------------------------------------------------


def test_repr_with_filter_expr():
    sr = _make_result([("A", 2, {"beta": 0.5}), ("B", 3, {"beta": 0.8})])
    filtered = sr.filter(sr.P.beta == 0.5)
    r = repr(filtered)
    assert "filter=" in r
    assert "beta" in r


def test_repr_no_filter_expr():
    sr = _make_result([("A", 2, {"beta": 0.5})])
    # No filter applied
    assert "filter=" not in repr(sr)
    # Lambda filter has no expr
    filtered = sr.filter(lambda pars, meta: True)
    assert "filter=" not in repr(filtered)


# ---------------------------------------------------------------------------
# Groupby tests
# ---------------------------------------------------------------------------


def test_groupby_single_key():
    sr = _make_result([("A", 2, {"beta": 0.5}), ("B", 3, {"beta": 0.8})])
    groups = list(sr.groupby("beta"))
    # sorted by key
    assert groups[0][0] == 0.5
    assert len(groups[0][1]) == 2
    assert groups[1][0] == 0.8
    assert len(groups[1][1]) == 3
    # Values are Results instances
    assert isinstance(groups[0][1], Results)


def test_groupby_multi_key():
    sr = _make_result([("A", 1, {"beta": 0.5, "alpha": 0.1}), ("B", 1, {"beta": 0.8, "alpha": 0.2})])
    groups = list(sr.groupby("beta", "alpha"))
    assert all(isinstance(k, tuple) and len(k) == 2 for k, _ in groups)
    keys = [k for k, _ in groups]
    assert (0.5, 0.1) in keys
    assert (0.8, 0.2) in keys


def test_groupby_P_field():
    sr = _make_result([("A", 2, {"beta": 0.5}), ("B", 3, {"beta": 0.8})])
    groups = list(sr.groupby(sr.P.beta))
    assert len(groups) == 2
    assert groups[0][0] == 0.5
    assert groups[1][0] == 0.8


def test_groupby_skips_empty():
    sr = _make_result([("A", 2, {"beta": 0.5}), ("B", 3, {"beta": 0.8})])
    # Filter first to ensure only one group remains
    filtered = sr.filter(sr.P.beta == 0.5)
    groups = list(filtered.groupby("beta"))
    assert len(groups) == 1
    assert groups[0][0] == 0.5


def test_groupby_is_iterator():
    sr = _make_result([("A", 2, {"beta": 0.5})])
    result = sr.groupby("beta")
    # It should be an iterator/generator, not a list or dict
    import types

    assert isinstance(result, types.GeneratorType)


# ---------------------------------------------------------------------------
# Navigation tests
# ---------------------------------------------------------------------------


def test_attr_study_access():
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        {
            "parameter_set": ps,
            "branch": "Control",
            "sim_id": i + 1,
            "branch_id": (i % 2) + 1,
            "pars": objdict({}),
        }
        for i, ps in enumerate(["StudyA", "StudyA", "StudyB", "StudyB"])
    ]
    sr = Results(_values=(metadata_list, values))
    assert len(sr.StudyA) == 2
    assert len(sr.StudyB) == 2


def test_attr_branch_after_study():
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        {
            "parameter_set": ps,
            "branch": br,
            "sim_id": i + 1,
            "branch_id": 1,
            "pars": objdict({}),
        }
        for i, (ps, br) in enumerate([("StudyA", "Arm1"), ("StudyA", "Arm2"), ("StudyB", "Arm1"), ("StudyB", "Arm2")])
    ]
    sr = Results(_values=(metadata_list, values))
    # Chain: study then branch
    chained = sr.StudyA.Arm1
    assert len(chained) == 1


def test_attr_study_wins_collision():
    """When name matches both study and branch, study wins silently."""
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        # Two different parameter_sets, one branch named "Baseline" in each
        {"parameter_set": "Baseline", "branch": "Baseline", "sim_id": 1, "branch_id": 1, "pars": objdict({})},
        {"parameter_set": "Baseline", "branch": "Treatment", "sim_id": 2, "branch_id": 1, "pars": objdict({})},
        {"parameter_set": "Other", "branch": "Baseline", "sim_id": 3, "branch_id": 1, "pars": objdict({})},
        {"parameter_set": "Other", "branch": "Control", "sim_id": 4, "branch_id": 1, "pars": objdict({})},
    ]
    sr = Results(_values=(metadata_list, values))
    # "Baseline" matches both a parameter_set and a branch name;
    # with multiple parameter_sets, study wins
    result = sr.Baseline
    # Should be filtered by parameter_set == "Baseline", giving 2 entries
    assert len(result) == 2
    for _pars, meta in result.iter_params():
        assert meta["parameter_set"] == "Baseline"


def test_studies_property():
    values = [{"v": i} for i in range(4)]
    metadata_list = [
        {
            "parameter_set": ps,
            "branch": "Control",
            "sim_id": i + 1,
            "branch_id": 1,
            "pars": objdict({}),
        }
        for i, ps in enumerate(["Alpha", "Alpha", "Beta", "Beta"])
    ]
    sr = Results(_values=(metadata_list, values))
    studies = sr.studies
    assert isinstance(studies, objdict)
    assert set(studies.keys()) == {"Alpha", "Beta"}
    assert isinstance(studies["Alpha"], Results)
    assert len(studies["Alpha"]) == 2
    assert len(studies["Beta"]) == 2


# ---------------------------------------------------------------------------
# Iteration tests
# ---------------------------------------------------------------------------


def test_items():
    sr = _make_result([("Baseline", 2, {"beta": 0.5})])
    items = list(sr.items())
    assert len(items) == 2
    (pars, meta), val = items[0]
    # pars is the parameter dict
    assert pars["beta"] == 0.5
    # metadata does NOT contain a 'pars' key
    assert "pars" not in meta
    assert "branch" in meta
    # value is loaded
    assert val == {"value": 1}


def test_iter_params():
    sr = _make_result([("Baseline", 2, {"beta": 0.5})])
    param_pairs = list(sr.iter_params())
    assert len(param_pairs) == 2
    for pars, meta in param_pairs:
        assert pars["beta"] == 0.5
        assert "pars" not in meta
        assert "branch" in meta


def test_first():
    sr = _make_result([("Baseline", 3, {"beta": 0.5})])
    (pars, meta), val = sr.first()
    assert pars["beta"] == 0.5
    assert meta["branch"] == "Baseline"
    assert val == {"value": 1}


def test_first_empty_raises():
    sr = Results([])
    with pytest.raises(IndexError):
        sr.first()


def test_metadata_split():
    """metadata from iter_params/items has no 'pars' key; pars is the param dict."""
    sr = _make_result([("A", 1, {"alpha": 0.1, "beta": 0.5})])
    (pars, meta), _val = sr.first()
    assert "pars" not in meta
    assert "alpha" in pars
    assert "beta" in pars
    assert meta["branch"] == "A"


# ---------------------------------------------------------------------------
# Display tests
# ---------------------------------------------------------------------------


def test_repr_html():
    sr = _make_result([("Baseline", 2, {"beta": 0.5}), ("Treatment", 3, {"beta": 0.8})])
    html = sr._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0
    # Contains the entry count
    assert "5" in html


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


def test_constructor_path(tmp_path):
    cache = SimFileCache(tmp_path, save=_pkl_save, load=_pkl_load)
    for i, key in enumerate(["key001", "key002"], start=1):
        cache.save(key, {"result": i}, {"branch": "B", "pars": {"x": i}})

    sr = Results(str(tmp_path))
    assert len(sr) == 2


def test_constructor_backend(tmp_path):
    cache = SimFileCache(tmp_path, save=_pkl_save, load=_pkl_load)
    for i, key in enumerate(["key001", "key002"], start=1):
        cache.save(key, {"result": i}, {"branch": "B", "pars": {"x": i}})

    sr = Results(cache)
    assert len(sr) == 2


def test_constructor_values():
    meta_list = [
        {"parameter_set": "s", "branch": "B", "sim_id": 1, "branch_id": 1, "pars": objdict({"x": 1})},
        {"parameter_set": "s", "branch": "B", "sim_id": 2, "branch_id": 2, "pars": objdict({"x": 2})},
    ]
    val_list = [{"v": 1}, {"v": 2}]
    sr = Results(_values=(meta_list, val_list))
    assert len(sr) == 2


def test_constructor_empty():
    sr = Results()
    assert len(sr) == 0
    assert not sr


def test_getitem_int_raises():
    sr = _make_result([("A", 3, {"x": 1})])
    with pytest.raises(TypeError):
        _ = sr[0]


# ---------------------------------------------------------------------------
# Missing attribute still raises
# ---------------------------------------------------------------------------


def test_attr_missing_raises():
    sr = _make_result([("Baseline", 2, {})])
    with pytest.raises(AttributeError, match="no attribute 'NonExistent'"):
        _ = sr.NonExistent
