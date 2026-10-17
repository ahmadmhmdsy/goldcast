import numpy as np

from goldcast.evaluation import metrics as M


def test_holm_matches_hand_calculation():
    adj = M.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": np.nan})
    assert np.isclose(adj["a"], 0.03)
    assert np.isclose(adj["c"], 0.06)
    assert np.isclose(adj["b"], 0.06)  # monotone: never below the previous step
    assert np.isnan(adj["d"])


def test_dm_detects_a_clearly_better_model_and_not_an_equal_one():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 500)
    good = (y - 0.8 * y) ** 2
    _, p = M.diebold_mariano(y**2, good)
    assert p < 1e-6
    _, p_same = M.diebold_mariano(y**2, y**2 + rng.normal(0, 1e-3, 500))
    assert p_same > 0.01


def test_pinball_known_value():
    loss = M.pinball(np.array([1.0]), np.array([[0.0, 1.0, 2.0]]), [0.1, 0.5, 0.9])
    # q=0.1 under-forecast by 1 -> 0.1; q=0.5 exact -> 0; q=0.9 over by 1 -> 0.1
    assert np.isclose(loss[0], 0.2 / 3)


def test_pt_rejects_skill_and_is_nan_for_constant_forecasts():
    rng = np.random.default_rng(2)
    up = rng.random(400) > 0.45
    skilled = np.where(rng.random(400) < 0.8, up, ~up)
    assert M.pesaran_timmermann(up, skilled)[1] < 1e-6
    assert np.isnan(M.pesaran_timmermann(up, np.ones(400, bool))[1])


def test_long_flat_charges_costs():
    r = M.long_flat(np.array([0.01, 0.01, 0.01]), np.array([1, 0, 1]), cost_bp=10, periods_per_year=52)
    assert np.isclose(r["strategy"]["total_logret"], 0.02 - 3 * 0.001)
    assert r["trades"] == 3
