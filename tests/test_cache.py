import pickle
from pathlib import Path

import numpy as np
import pytest

from parsimmon.cache import SimCacheBase, SimFileCache

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestSimFileCache:
    def test_save_load_roundtrip(self, cache):
        key = "abc123def456abcd"
        data = {"x": np.array([1.0, 2.0, 3.0]), "y": [10, 20], "z": "hello"}
        cache.save(key, data, metadata={"label": "test"})
        loaded = cache.load(key)
        assert loaded["y"] == data["y"]
        assert loaded["z"] == data["z"]
        assert np.array_equal(loaded["x"], data["x"])

    def test_exists(self, cache):
        key = "aabbccddeeff0011"
        assert not cache.exists(key)
        cache.save(key, {"val": 42}, metadata={})
        assert cache.exists(key)

    def test_index(self, cache):
        key = "1122334455667788"
        meta = {"label": "run_1", "fn_hash": "deadbeef12345678"}
        cache.save(key, [1, 2, 3], metadata=meta)
        entries = cache.index()
        assert len(entries) == 1
        entry = entries[0]
        assert entry["cache_key"] == key
        assert entry["label"] == "run_1"
        assert entry["fn_hash"] == "deadbeef12345678"
        assert "timestamp" in entry

    def test_add_index_entry(self, cache):
        key = "ffffffffffffffff"
        cache.add_index_entry({"cache_key": key, "label": "dedup_hit", "source": "cross-set"})
        assert not cache.exists(key)
        entries = cache.index()
        assert len(entries) == 1
        assert entries[0]["label"] == "dedup_hit"

    def test_keys(self, cache):
        assert cache.keys() == []
        cache.save("aaaa000011110000", {"a": 1}, metadata={})
        cache.save("bbbb111122220000", {"b": 2}, metadata={})
        assert sorted(cache.keys()) == sorted(["aaaa000011110000", "bbbb111122220000"])

    def test_delete(self, cache):
        key = "deadbeefcafe0000"
        cache.save(key, {"val": 99}, metadata={"label": "to_delete"})
        assert cache.exists(key)
        cache.delete(key)
        assert not cache.exists(key)
        assert all(e.get("cache_key") != key for e in cache.index())

    def test_clear(self, cache):
        for i, key in enumerate(["aaaa0000aaaa0000", "bbbb1111bbbb1111", "cccc2222cccc2222"]):
            cache.save(key, {"i": i}, metadata={})
        assert len(cache.keys()) == 3
        cache.clear()
        assert cache.keys() == []
        assert cache.index() == []

    def test_idempotent_save(self, tmp_path):
        saved_calls = []

        def tracking_save(path, obj):
            saved_calls.append(path)
            with open(path, "wb") as f:
                pickle.dump(obj, f)

        def tracking_load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        cache = SimFileCache(tmp_path / "cache", save=tracking_save, load=tracking_load)
        key = "idempotent000000"
        cache.save(key, {"v": 1}, metadata={"run": 1})
        cache.save(key, {"v": 99}, metadata={"run": 2})

        result_path = str(tmp_path / "cache" / "results" / f"{key}.pkl")
        result_writes = [p for p in saved_calls if result_path in str(p)]
        assert len(result_writes) == 1
        assert cache.load(key) == {"v": 1}

    def test_index_persistence(self, tmp_path):
        cache_dir = tmp_path / "persistent_cache"
        key = "persist000000000"
        SimFileCache(cache_dir).save(key, {"data": "original"}, metadata={"label": "persisted"})

        cache2 = SimFileCache(cache_dir)
        entries = cache2.index()
        assert len(entries) == 1
        assert entries[0]["cache_key"] == key
        assert entries[0]["label"] == "persisted"
        assert cache2.exists(key)

    def test_get_fn_hash(self, cache):
        key = "fnhash0000000000"
        fn_hash = "abcd1234efgh5678"
        cache.save(key, {"result": 42}, metadata={"fn_hash": fn_hash, "label": "v1"})
        assert cache.get_fn_hash(key) == fn_hash
        assert cache.get_fn_hash("nonexistent000000") is None

    def test_certify_entries(self, cache):
        old_hash = "oldhash000000000"
        new_hash = "newhash111111111"
        for i, key in enumerate(["aaaa0000aaaa0001", "aaaa0000aaaa0002", "aaaa0000aaaa0003"]):
            cache.save(key, {"i": i}, metadata={"fn_hash": old_hash, "label": f"run_{i}"})
        assert cache.certify_entries(new_hash) == 3
        assert all(e["fn_hash"] == new_hash for e in cache.index())

    def test_certify_entries_filtered(self, cache):
        old_hash, new_hash = "oldhash000000000", "newhash111111111"
        cache.save("aaaa0000aaaa0011", {"v": 1}, metadata={"fn_hash": old_hash, "parameter_set": "set_a"})
        cache.save("aaaa0000aaaa0012", {"v": 2}, metadata={"fn_hash": old_hash, "parameter_set": "set_a"})
        cache.save("bbbb1111bbbb0011", {"v": 3}, metadata={"fn_hash": old_hash, "parameter_set": "set_b"})

        assert cache.certify_entries(new_hash, parameter_set="set_a") == 2
        for entry in cache.index():
            expected = new_hash if entry.get("parameter_set") == "set_a" else old_hash
            assert entry["fn_hash"] == expected

    def test_clean_entries_stale(self, cache):
        current_hash, stale_hash = "currenthash11111", "stalehash0000000"
        cache.save("stale00000000001", {"v": 1}, metadata={"fn_hash": stale_hash})
        cache.save("stale00000000002", {"v": 2}, metadata={"fn_hash": stale_hash})
        cache.save("live000000000001", {"v": 3}, metadata={"fn_hash": current_hash})

        assert cache.clean_entries(current_fn_hash=current_hash) == 2
        remaining = cache.index()
        assert len(remaining) == 1
        assert remaining[0]["cache_key"] == "live000000000001"

    def test_clean_entries_all(self, cache):
        keys = ["cccc0000cccc0001", "cccc0000cccc0002", "cccc0000cccc0003"]
        for i, key in enumerate(keys):
            cache.save(key, {"i": i}, metadata={"fn_hash": "somehash00000000"})
        assert cache.clean_entries(remove_all=True) == 3
        assert cache.index() == []

    def test_remove_orphans(self, cache):
        cache.save("indexed000000001", {"v": 1}, metadata={"fn_hash": "somehash00000001"})

        cache._ensure_dirs()
        (cache._results_dir / "orphan0000000001.pkl").write_bytes(b"orphan data 1")
        (cache._results_dir / "orphan0000000002.pkl").write_bytes(b"orphan data 2")

        assert cache.remove_orphans() == 2
        assert not (cache._results_dir / "orphan0000000001.pkl").exists()
        assert (cache._results_dir / "indexed000000001.pkl").exists()


def test_base_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        SimCacheBase()
