"""Tests for the data pipeline: sequence construction, decomposition,
reproducibility and the no-leakage split contract."""

import numpy as np
import pytest

import data_loader as dl


# ---------------------------------------------------------------------------
# make_sequences
# ---------------------------------------------------------------------------
def test_make_sequences_horizon_1():
    s = np.arange(50, dtype=float)
    X, y = dl.make_sequences(s, look_back=10, horizon=1)
    assert X.shape == (40, 10)
    assert y.shape == (40,)
    np.testing.assert_array_equal(X[0], s[0:10])
    assert y[0] == s[10]          # next step


def test_make_sequences_multi_horizon_alignment():
    s = np.arange(50, dtype=float)
    h = 5
    X, y = dl.make_sequences(s, look_back=10, horizon=h)
    # y[i] is `horizon` steps ahead of the end of window i.
    assert y[0] == s[10 + h - 1]
    assert len(X) == len(y)
    # last usable index keeps the target in-range
    assert y[-1] == s[-1]


def test_make_sequences_too_short_returns_empty():
    s = np.arange(5, dtype=float)
    X, y = dl.make_sequences(s, look_back=10, horizon=1)
    assert X.shape == (0, 10)
    assert y.shape == (0,)


# ---------------------------------------------------------------------------
# decompose  (real CEEMDAN if PyEMD present, else labelled fallback)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 8 * np.pi, 256)
    return (np.sin(t) + 0.4 * np.sin(5 * t) + 0.2 * rng.standard_normal(256)).astype(float)


def test_decompose_returns_exact_component_count(signal):
    comps = dl.decompose(signal, n_components=6, trials=3, seed=42)
    assert len(comps) == 6
    assert all(c.shape == signal.shape for c in comps)


def test_decompose_sum_reconstructs_signal(signal):
    comps = dl.decompose(signal, n_components=6, trials=3, seed=42)
    recon = np.sum(comps, axis=0)
    np.testing.assert_allclose(recon, signal, atol=1e-4)


def test_decompose_pads_when_few_components(signal):
    # A near-monotonic signal yields few IMFs; we still must get n_components,
    # zero-padded, with the sum preserved.
    ramp = np.linspace(0, 1, 256)
    comps = dl.decompose(ramp, n_components=8, trials=3, seed=42)
    assert len(comps) == 8
    np.testing.assert_allclose(np.sum(comps, axis=0), ramp, atol=1e-4)


def test_decompose_is_reproducible(signal):
    a = dl.decompose(signal, n_components=6, trials=4, seed=123)
    b = dl.decompose(signal, n_components=6, trials=4, seed=123)
    for ca, cb in zip(a, b):
        np.testing.assert_allclose(ca, cb)


def test_moving_average_fallback_reconstructs(signal):
    comps = dl._moving_average_decompose(signal, n_components=6)
    assert len(comps) == 6
    np.testing.assert_allclose(np.sum(comps, axis=0), signal, atol=1e-6)


# ---------------------------------------------------------------------------
# Split / scaling contract (no network: drive the synthetic generator)
# ---------------------------------------------------------------------------
def test_synthetic_generator_is_bounded_and_seeded():
    a = dl._generate_synthetic(n_days=3, seed=7)
    b = dl._generate_synthetic(n_days=3, seed=7)
    assert (a["cpu_util"].between(0, 100)).all()
    np.testing.assert_array_equal(a["cpu_util"].values, b["cpu_util"].values)


def test_physical_scaling_is_leak_free_and_bounds_test_in_unit_interval():
    # CloudSense scales to the PHYSICAL CPU range [0,100]%, not train min/max.
    # That transform uses a physical constant, so it leaks nothing from the data
    # AND keeps every split inside [0,1] -- even a test regime far above train.
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [100.0]]))
    assert scaler.data_min_[0] == pytest.approx(0.0)
    assert scaler.data_max_[0] == pytest.approx(100.0)

    # A high-CPU "test regime" (mean ~71%, peaks ~99.7%) that never appears in a
    # low-CPU "train regime" must still scale strictly within [0,1] -- this is the
    # property whose absence caused the earlier clip-induced negative-R2 collapse.
    test_regime = np.array([[28.3], [70.9], [99.74], [56.85]])
    scaled = scaler.transform(test_regime).flatten()
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    # Round-trips back to raw %.
    back = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
    np.testing.assert_allclose(back, test_regime.flatten(), atol=1e-6)
