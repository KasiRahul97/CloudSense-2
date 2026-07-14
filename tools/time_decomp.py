"""Time per-window CEEMDAN so we can size the causal-decomposition full run.

Run as a FILE:  PYTHONPATH=src .venv/Scripts/python.exe tools/time_decomp.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_loader as dl  # noqa: E402

rng = np.random.default_rng(0)


def bench(win_len, trials, n_comp, reps=20):
    # Representative-ish CPU window: slow trend + diurnal + noise.
    t = np.linspace(0, win_len / 12.0, win_len)  # ~5-min steps
    base = 40 + 20 * np.sin(2 * np.pi * t / 24.0) + 8 * np.sin(2 * np.pi * t / 2.0)
    sigs = [np.clip(base + 5 * rng.standard_normal(win_len), 0, 100) for _ in range(reps)]
    t0 = time.perf_counter()
    n_imfs = []
    for s in sigs:
        comps = dl.decompose(s, n_components=n_comp, trials=trials, seed=42)
        nz = sum(1 for c in comps if np.any(np.abs(c) > 1e-9))
        n_imfs.append(nz)
    dt = (time.perf_counter() - t0) / reps
    print(f"  win={win_len:4d} trials={trials:2d} -> {dt*1000:7.1f} ms/window "
          f"| non-zero comps median={int(np.median(n_imfs))}/{n_comp}")
    return dt


def main():
    print(f"CEEMDAN available: {dl.ceemdan_available()}")
    n_comp = 8
    for win in (48, 96, 192, 288):
        for trials in (8, 20):
            dt = bench(win, trials, n_comp)
            # Project full-run cost: ~2775 train + 355 val + 758 test windows.
            n_windows = 2775 + 355 + 758
            print(f"        => ~{dt * n_windows/60:5.1f} min for {n_windows} windows "
                  f"(one split-set, one horizon)")


if __name__ == "__main__":
    main()
