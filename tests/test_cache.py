"""Cache write/read round trip (NumPy only, runs on the laptop)."""
import numpy as np

from minivla import cache


def test_round_trip_with_text_padding(tmp_path):
    rng = np.random.default_rng(0)
    arrays = cache.create(tmp_path, n_rows=5, n_prefix=3, n_text=4, dim=2, meta={"note": "toy"})
    p1, t1 = rng.standard_normal((2, 3, 2)), rng.standard_normal((2, 4, 2))  # full-length text
    p2, t2 = rng.standard_normal((3, 3, 2)), rng.standard_normal((3, 2, 2))  # shorter text
    cache.write_rows(arrays, 0, p1, t1, np.ones((2, 4), bool), [10, 11])
    cache.write_rows(arrays, 2, p2, t2, np.ones((3, 2), bool), [12, 13, 14])
    for a in arrays.values():
        a.flush()
    del arrays

    meta, c = cache.open_cache(tmp_path)
    assert meta["note"] == "toy" and meta["n_rows"] == 5 and meta["n_text"] == 4
    np.testing.assert_array_equal(c["prefix"][:2], p1.astype(np.float16))
    np.testing.assert_array_equal(c["text"][2:, :2], t2.astype(np.float16))
    assert (c["text"][2:, 2:] == 0).all()
    assert c["text_mask"][2:].sum(1).tolist() == [2, 2, 2]
    assert c["frame_index"].tolist() == [10, 11, 12, 13, 14]


def test_size_gb():
    assert cache.size_gb(n_rows=1000, n_tokens=10, dim=100) == 1000 * 10 * 100 * 2 / 1e9
