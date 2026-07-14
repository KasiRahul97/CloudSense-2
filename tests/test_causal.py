"""Tests for the causal, leakage-free feature pipeline (RevIN + per-window
CEEMDAN) shared by training and serving."""

import numpy as np
import pytest

import data_loader as dl


LB = 48
NC = 6
TR = 3


@pytest.fixture(scope="module")
def window():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 6, LB)
    # scaled [0,1] window with structure (a 4-hour CPU window / 100)
    return np.clip(0.45 + 0.2 * np.sin(t) + 0.03 * rng.standard_normal(LB), 0, 1)


# ---------------------------------------------------------------------------
# revin_stats
# ---------------------------------------------------------------------------
def test_revin_stats_matches_mean_std(window):
    mu, sd = dl.revin_stats(window)
    assert mu == pytest.approx(float(window.mean()))
    assert sd == pytest.approx(float(window.std()))


def test_revin_stats_floors_std_on_flat_window():
    flat = np.full(LB, 0.5)
    mu, sd = dl.revin_stats(flat, eps=1e-2)
    assert mu == pytest.approx(0.5)
    assert sd == 1e-2                      # floored, never zero


# ---------------------------------------------------------------------------
# window_components  (causal per-window decomposition of the RevIN signal)
# ---------------------------------------------------------------------------
def test_window_components_shape_and_reconstruction(window):
    mu, sd = dl.revin_stats(window)
    comps = dl.window_components(window, mu, sd, n_components=NC, trials=TR, seed=42)
    assert comps.shape == (NC, LB)
    # The components are an additive decomposition of the RevIN-normalized window.
    norm = (window - mu) / sd
    np.testing.assert_allclose(comps.sum(axis=0), norm, atol=1e-4)


def test_window_components_reproducible(window):
    mu, sd = dl.revin_stats(window)
    a = dl.window_components(window, mu, sd, NC, TR, 42)
    b = dl.window_components(window, mu, sd, NC, TR, 42)
    np.testing.assert_allclose(a, b)


def test_window_components_flat_window_is_finite_and_in_residue():
    flat = np.full(LB, 0.5)
    mu, sd = dl.revin_stats(flat)
    comps = dl.window_components(flat, mu, sd, NC, TR, 42)
    assert comps.shape == (NC, LB)
    assert np.isfinite(comps).all()          # no CEEMDAN divide-by-zero NaNs
    # Flat (normalized to ~0) window: all components are ~0.
    np.testing.assert_allclose(comps.sum(axis=0), (flat - mu) / sd, atol=1e-6)


# ---------------------------------------------------------------------------
# features_from_raw_pct  (serving-side; must mirror training)
# ---------------------------------------------------------------------------
def test_features_from_raw_pct_shapes_and_window_use():
    rng = np.random.default_rng(1)
    raw = np.clip(50 + 15 * np.sin(np.linspace(0, 8, LB + 20)) + rng.normal(0, 3, LB + 20), 0, 100)
    comps, mu, sd = dl.features_from_raw_pct(raw, look_back=LB, n_components=NC,
                                             trials=TR, seed=42)
    assert comps.shape == (NC, LB)
    # mu/sd are computed on the SCALED last-look_back window.
    last_scaled = raw[-LB:] / 100.0
    assert mu == pytest.approx(float(last_scaled.mean()), abs=1e-6)


def test_features_physical_scaler_equals_divide_by_100():
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(np.array([[0.0], [100.0]]))
    rng = np.random.default_rng(2)
    raw = np.clip(40 + 20 * np.sin(np.linspace(0, 6, LB)) + rng.normal(0, 2, LB), 0, 100)
    c1, m1, s1 = dl.features_from_raw_pct(raw, LB, NC, TR, 42, scaler=scaler)
    c2, m2, s2 = dl.features_from_raw_pct(raw, LB, NC, TR, 42, scaler=None)
    np.testing.assert_allclose(c1, c2, atol=1e-6)
    assert (m1, s1) == pytest.approx((m2, s2))


def test_features_rejects_short_window():
    with pytest.raises(ValueError):
        dl.features_from_raw_pct([50.0] * (LB - 1), LB, NC, TR, 42)
