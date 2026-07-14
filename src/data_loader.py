"""
data_loader.py
==============
Dataset loading, splitting, scaling, sequence construction and signal
decomposition for the CloudSense workload-forecasting benchmark.

Design notes (read before changing anything)
---------------------------------------------
1. SINGLE EC2 INSTANCE.
   We model ONE Numenta NAB EC2 CPU series, not a concatenation of several.
   Concatenating heterogeneous instances and then doing a temporal split puts
   instances with very different CPU distributions in train vs. test, which
   creates a large train/test distribution shift. That shift is what made the
   earlier baselines collapse to negative R^2. A single instance is the
   standard, clean setup for NAB CPU forecasting.

2. LEAKAGE-AWARE DECOMPOSITION.
   The proposed model decomposes the signal into components. EMD-family methods
   are *non-causal* (the component value at time t depends on the whole signal,
   including the future). To avoid the most egregious leakage we decompose the
   train series and the test series as SEPARATE blocks, so the training
   decomposition never sees test data. A fully causal rolling decomposition is
   left as future work and is documented as a known limitation in the README.
   The deployed API (deploy/inference/app.py) decomposes only the supplied
   past window, which IS causal.

3. DIRECT MULTI-HORIZON FORECASTING.
   make_sequences(series, look_back, horizon) builds targets `horizon` steps
   ahead. horizon=1 -> next 5-min step; horizon=48 -> 4 hours ahead. This is
   what lets us benchmark a genuine multi-hour prediction horizon rather than
   conflating the look-back window with the horizon.
"""

from __future__ import annotations

import os
import logging
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("cloudsense.data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch"
# A single real EC2 CPU series with genuine 0-100% dynamics (14 days @ 5-min =
# 4032 pts; range ~2.5-99.7%, mean ~41%, std ~22%). Chosen over the near-idle
# NAB instances (e.g. 24ae8d, max 2.3%) so the forecasting task is non-trivial.
DEFAULT_NAB_FILE = "ec2_cpu_utilization_ac20cd.csv"

LOOK_BACK = 48     # 48 * 5-min = 4-hour input window
TEST_RATIO = 0.20
VAL_RATIO = 0.10
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed all RNGs we use so results are reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # torch not always present at import time
        pass


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
def _download_nab(save_dir: str, nab_file: str) -> Optional[str]:
    """Download a single NAB EC2 file; return its local path (or None)."""
    os.makedirs(save_dir, exist_ok=True)
    local = os.path.join(save_dir, nab_file)
    if os.path.exists(local):
        logger.info("Using cached NAB file %s", nab_file)
        return local
    url = f"{NAB_BASE}/{nab_file}"
    try:
        import requests
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        # Some networks (e.g. university SSL-inspection proxies) present a private
        # root CA that breaks certificate verification. Allow an explicit, logged
        # opt-out so the real dataset can still be fetched; default stays secure.
        if os.getenv("CLOUDSENSE_INSECURE_DOWNLOAD") == "1":
            try:
                import requests
                import urllib3
                urllib3.disable_warnings()
                logger.warning("Secure download failed (%s); retrying with TLS "
                               "verification DISABLED (CLOUDSENSE_INSECURE_DOWNLOAD=1).", e)
                resp = requests.get(url, timeout=20, verify=False)
                resp.raise_for_status()
            except Exception as e2:
                logger.warning("Insecure download also failed (%s); synthetic fallback.", e2)
                return None
        else:
            logger.warning("Could not download NAB data (%s); using synthetic fallback. "
                           "On a TLS-intercepting proxy, set CLOUDSENSE_INSECURE_DOWNLOAD=1.", e)
            return None
    with open(local, "w") as f:
        f.write(resp.text)
    logger.info("Downloaded NAB file %s", nab_file)
    return local


def _load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns={"value": "cpu_util"})
    df = df[["timestamp", "cpu_util"]].dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["cpu_util"] = df["cpu_util"].clip(0, 100)
    return df


def _generate_synthetic(n_days: int = 14, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Offline fallback ONLY. Realistic AWS-like EC2 workload with daily/weekly
    seasonality and occasional spikes. Clearly labelled as synthetic so it is
    never confused with the real NAB benchmark in reporting.
    """
    rng = np.random.default_rng(seed)
    n = n_days * 24 * 12  # 5-min intervals
    t = np.arange(n)
    h = (t * 5 / 60) % 24
    daily = 28 * np.exp(-0.5 * ((h - 10) / 2.2) ** 2) + 22 * np.exp(-0.5 * ((h - 15) / 1.8) ** 2)
    weekly = np.where(((t * 5 // (60 * 24)) % 7) >= 5, -15.0, 0.0)
    trend = 6.0 * t / n
    spikes = np.zeros(n)
    idx = np.where(rng.poisson(0.003, n) > 0)[0]
    for i in idx:
        d = min(int(rng.integers(2, 8)), n - i)
        spikes[i:i + d] += rng.uniform(12, 35) * np.exp(-0.3 * np.arange(d))
    noise = rng.normal(0, 2.5, n)
    cpu = np.clip(28 + daily + weekly + trend + spikes + noise, 2, 98)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "cpu_util": cpu.round(2)})


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
def ceemdan_available() -> bool:
    try:
        import PyEMD  # noqa: F401
        return True
    except Exception:
        return False


def decompose(signal: np.ndarray, n_components: int, trials: int = 20,
              seed: int = DEFAULT_SEED) -> List[np.ndarray]:
    """
    Decompose `signal` into exactly `n_components` additive components whose sum
    reconstructs the original signal.

    Uses real CEEMDAN (PyEMD) when available. Falls back to an explicitly
    labelled multi-scale moving-average decomposition only if PyEMD is missing
    (the fallback is NOT CEEMDAN and is reported as such).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if ceemdan_available():
        comps = _ceemdan_decompose(signal, n_components, trials=trials, seed=seed)
        method = "CEEMDAN"
    else:
        logger.warning("PyEMD not installed; using moving-average decomposition (NOT CEEMDAN).")
        comps = _moving_average_decompose(signal, n_components)
        method = "moving-average (fallback)"
    logger.debug("Decomposed signal (len=%d) into %d components via %s",
                 len(signal), len(comps), method)
    return comps


def _align_components(comps: List[np.ndarray], n_components: int,
                      n: int) -> List[np.ndarray]:
    """Force exactly n_components arrays of length n, preserving sum == signal."""
    comps = [np.asarray(c, dtype=np.float64).reshape(-1)[:n] for c in comps]
    if len(comps) > n_components:
        # Merge the extra low-frequency components into the last one.
        merged = np.sum(comps[n_components - 1:], axis=0)
        comps = comps[:n_components - 1] + [merged]
    while len(comps) < n_components:
        comps.append(np.zeros(n, dtype=np.float64))
    return comps


def _ceemdan_decompose(signal: np.ndarray, n_components: int, trials: int,
                       seed: int) -> List[np.ndarray]:
    from PyEMD import CEEMDAN
    # parallel=False is deliberate: PyEMD defaults to parallel=True, which spawns
    # worker processes that re-import the entry script. On Windows that is fragile
    # (and impossible when the caller runs from stdin). Serial CEEMDAN is reliable
    # and fast enough for our ~3k-point series. noise_seed() makes it reproducible.
    ceemdan = CEEMDAN(trials=trials, parallel=False)
    ceemdan.noise_seed(seed)
    # max_imf caps the IMF count; +residue gives up to n_components pieces.
    ceemdan.ceemdan(signal, max_imf=n_components - 1)
    imfs, residue = ceemdan.get_imfs_and_residue()
    comps = [imfs[i] for i in range(imfs.shape[0])] + [residue]
    return _align_components(comps, n_components, len(signal))


def _moving_average_decompose(signal: np.ndarray, n_components: int) -> List[np.ndarray]:
    """Labelled fallback: trailing multi-scale moving averages (causal)."""
    n = len(signal)
    residue = signal.copy()
    comps: List[np.ndarray] = []
    windows = [w for w in [4, 8, 16, 32, 64, 128, 256] if w < n][:n_components - 1]
    for w in windows:
        kernel = np.ones(w) / w
        smooth = np.convolve(residue, kernel, mode="full")[:n]  # trailing MA (causal)
        comps.append(residue - smooth)
        residue = smooth
    comps.append(residue)
    return _align_components(comps, n_components, n)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
def make_sequences(series: np.ndarray, look_back: int, horizon: int = 1):
    """
    Build (X, y) for DIRECT h-step-ahead forecasting.

    X[i] = series[i-look_back : i]      (window of `look_back` past values)
    y[i] = series[i + horizon - 1]      (value `horizon` steps ahead)
    """
    series = np.asarray(series, dtype=np.float32)
    X, y = [], []
    last = len(series) - horizon + 1
    for i in range(look_back, last):
        X.append(series[i - look_back:i])
        y.append(series[i + horizon - 1])
    if not X:
        return np.empty((0, look_back), np.float32), np.empty((0,), np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Causal per-window normalization (RevIN) + decomposition
# ---------------------------------------------------------------------------
# These three functions are the heart of the leakage-free, deployment-faithful
# pipeline and are shared verbatim between training (load_dataset) and serving
# (deploy/inference). Because they touch ONLY the supplied past window, the
# benchmark numbers are exactly what the deployed API can reproduce online.
EPS_SCALED = 1e-2          # std floor in [0,1] units (~1% CPU) for flat windows
DECOMP_CACHE_DIRNAME = ".decomp_cache"


def revin_stats(window_scaled: np.ndarray, eps: float = EPS_SCALED):
    """Per-window normalization stats (RevIN). Causal: uses only this window.

    Subtracting the window mean and dividing by its std makes the model
    LEVEL-INVARIANT, which is what lets a model trained on a low-CPU regime
    forecast a high-CPU regime (the train/test shift in this dataset). The eps
    floor keeps a flat window (std ~ 0) from exploding.
    """
    w = np.asarray(window_scaled, dtype=np.float64)
    return float(w.mean()), max(float(w.std()), eps)


def window_components(window_scaled: np.ndarray, mu: float, sd: float,
                      n_components: int, trials: int, seed: int) -> np.ndarray:
    """Causally decompose ONE window into n_components level-invariant pieces.

    RevIN-normalize the window with (mu, sd), then CEEMDAN-decompose it. The
    decomposition sees only the supplied past window (causal / deployable), and
    operates on the normalized signal so the components do not depend on the
    absolute CPU level. Returns (n_components, look_back) float32.
    """
    norm = (np.asarray(window_scaled, dtype=np.float64) - mu) / sd

    # CEEMDAN divides by the signal's amplitude, so a (near-)constant window
    # triggers a divide-by-zero and yields NaNs. A flat window carries no
    # oscillatory structure anyway, so route it entirely into the residue.
    def _flat_fallback():
        comps = [np.zeros_like(norm) for _ in range(n_components - 1)] + [norm.copy()]
        return np.stack([c.astype(np.float32) for c in comps], axis=0)

    if not np.isfinite(norm).all() or float(np.ptp(norm)) < 1e-9:
        return _flat_fallback()

    comps = decompose(norm, n_components=n_components, trials=trials, seed=seed)
    arr = np.stack([np.asarray(c, dtype=np.float32) for c in comps], axis=0)
    if not np.isfinite(arr).all():        # CEEMDAN still misbehaved -> safe fallback
        return _flat_fallback()
    return arr


def features_from_raw_pct(window_pct, look_back, n_components, trials, seed,
                          scaler=None):
    """Serving-side feature builder: raw-% window -> (components, mu, sd).

    Mirrors the training pipeline EXACTLY (same scaling, RevIN and decomposition),
    so deploy/inference produces the same features the benchmark trained on.
    `window_pct` must hold at least `look_back` recent CPU-% samples.
    """
    w = np.asarray(window_pct, dtype=np.float64).reshape(-1)[-look_back:]
    if len(w) < look_back:
        raise ValueError(f"need >= {look_back} samples, got {len(w)}")
    if scaler is not None:
        w_scaled = scaler.transform(w.reshape(-1, 1)).flatten()
    else:
        w_scaled = w / 100.0
    mu, sd = revin_stats(w_scaled)
    comps = window_components(w_scaled, mu, sd, n_components, trials, seed)
    return comps, mu, sd


def _build_or_load_components(scaled, origins, look_back, n_components, trials,
                              seed, cache_dir, tag, verbose=True):
    """Decompose every causal window once (origin-keyed, horizon-independent),
    caching to disk so repeat runs and the second horizon are instant."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, tag + ".npz")
    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path)
            if (z["comps"].shape == (len(origins), n_components, look_back)
                    and np.array_equal(z["origins"], origins)):
                if verbose:
                    print(f"    [cache] components <- {os.path.basename(cache_path)}",
                          flush=True)
                return z["comps"], z["mus"], z["sds"]
        except Exception:
            pass  # corrupt/stale cache -> recompute

    if verbose:
        print(f"    decomposing {len(origins)} causal windows "
              f"(one-time, then cached) ...", flush=True)
    comps = np.zeros((len(origins), n_components, look_back), dtype=np.float32)
    mus = np.zeros(len(origins), dtype=np.float32)
    sds = np.zeros(len(origins), dtype=np.float32)
    for i, o in enumerate(origins):
        w = scaled[o - look_back:o]
        mu, sd = revin_stats(w)
        mus[i], sds[i] = mu, sd
        comps[i] = window_components(w, mu, sd, n_components, trials, seed)
        if verbose and (i + 1) % 500 == 0:
            print(f"      {i + 1}/{len(origins)} windows", flush=True)
    np.savez_compressed(cache_path, comps=comps, mus=mus, sds=sds,
                        origins=origins)
    return comps, mus, sds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def load_dataset(data_dir: str = "data", look_back: int = LOOK_BACK,
                 horizon: int = 1, n_components: int = 8, trials: int = 20,
                 nab_file: str = DEFAULT_NAB_FILE, seed: int = DEFAULT_SEED,
                 verbose: bool = True) -> dict:
    """
    Load one NAB EC2 CPU series and build a LEAKAGE-FREE, CAUSAL dataset:

      * scale to the physical CPU range [0, 100]% (a constant -> no leakage);
      * for every forecast origin, RevIN-normalize the look_back window and
        CEEMDAN-decompose it (causal: only past data, matches the served API);
      * assign each example to train/val/test by the split its TARGET falls in.

    All models (naive, learned baselines and the proposed ensemble) are scored
    against the same per-origin targets, so the comparison is apples-to-apples
    AND reproducible online by deploy/inference. Returns a dict for
    train_evaluate.py. Decomposition is cached on disk per (file, look_back,
    n_components, trials, seed) and reused across horizons.
    """
    set_seed(seed)

    path = _download_nab(data_dir, nab_file)
    df = _load_series(path) if path else None
    is_synthetic = False
    if df is None or len(df) < 500:
        df = _generate_synthetic(seed=seed)
        is_synthetic = True

    logger.info("Dataset: %d samples | mean=%.1f%% std=%.1f%% min=%.1f%% max=%.1f%%%s",
                len(df), df.cpu_util.mean(), df.cpu_util.std(),
                df.cpu_util.min(), df.cpu_util.max(),
                "  [SYNTHETIC FALLBACK]" if is_synthetic else "")

    raw_pct = df["cpu_util"].values.astype(np.float64)
    n = len(raw_pct)

    # Physical [0,100]% scaling (constant transform -> zero data leakage, and it
    # never sends a value outside [0,1]; see the regime-shift note in the README).
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [100.0]], dtype=np.float64))
    scaled = scaler.transform(raw_pct.reshape(-1, 1)).flatten()

    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    # Per-window causal features (decomposition is horizon-independent -> cache).
    origins = np.arange(look_back, n)            # window = scaled[o-look_back:o]
    tag = (f"{os.path.splitext(nab_file)[0]}_lb{look_back}_nc{n_components}"
           f"_tr{trials}_sd{seed}{'_syn' if is_synthetic else ''}")
    comps, mus, sds = _build_or_load_components(
        scaled, origins, look_back, n_components, trials, seed,
        cache_dir=os.path.join(data_dir, DECOMP_CACHE_DIRNAME), tag=tag,
        verbose=verbose)
    X_scaled_all = np.stack([scaled[o - look_back:o] for o in origins]).astype(np.float32)

    # Group examples by the split their TARGET (o + horizon - 1) lands in.
    splits = {"train": [], "val": [], "test": []}
    for idx, o in enumerate(origins):
        t = o + horizon - 1
        if t >= n:
            continue
        if t < n_train:
            splits["train"].append(idx)
        elif t < n_train + n_val:
            splits["val"].append(idx)
        else:
            splits["test"].append(idx)

    out_splits = {}
    for s, idxs in splits.items():
        idxs = np.array(idxs, dtype=int)
        tgt = origins[idxs] + horizon - 1
        out_splits[s] = {
            "X_scaled": X_scaled_all[idxs],                 # (N, look_back)
            "Xc": comps[idxs],                              # (N, n_components, look_back)
            "mu": mus[idxs].astype(np.float32),             # (N,)
            "sd": sds[idxs].astype(np.float32),             # (N,)
            "y_scaled": scaled[tgt].astype(np.float32),     # (N,)
        }

    return {
        "scaler": scaler,
        "raw_df": df,
        "is_synthetic": is_synthetic,
        "scaled_series": scaled,
        "look_back": look_back, "horizon": horizon,
        "n_components": n_components, "trials": trials, "seed": seed,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "splits": out_splits,
        "normalization": "revin_window_meanstd",
        "eps_scaled": EPS_SCALED,
        "decomposition": "CEEMDAN" if ceemdan_available() else "moving-average(fallback)",
    }
