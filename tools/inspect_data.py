"""Diagnostic: understand the chosen NAB series and why naive persistence scores
the way it does. Run as a FILE (not via stdin) so no multiprocessing re-import:

    PYTHONPATH=src .venv/Scripts/python.exe tools/inspect_data.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_loader as dl  # noqa: E402


def seg_stats(name, a):
    print(f"  {name:<6} n={len(a):>5}  min={a.min():6.2f} max={a.max():6.2f} "
          f"mean={a.mean():6.2f} std={a.std():6.2f}")


def persistence_r2(series, horizon):
    """Raw-units naive persistence on `series`: predict s[t]=s[t-horizon]."""
    y_true = series[horizon:]
    y_pred = series[:-horizon]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(y_true - y_pred))
    return mae, r2


def main():
    data_dir = os.path.join(ROOT, "data")
    csv = os.path.join(data_dir, dl.DEFAULT_NAB_FILE)
    df = pd.read_csv(csv)
    s = df["value"].values.astype(float)
    print(f"FILE: {dl.DEFAULT_NAB_FILE}")
    print(f"  columns: {list(df.columns)}")
    seg_stats("FULL", s)

    n = len(s)
    n_test = int(n * dl.TEST_RATIO)
    n_val = int(n * dl.VAL_RATIO)
    n_train = n - n_val - n_test
    train, val, test = s[:n_train], s[n_train:n_train + n_val], s[n_train + n_val:]
    print(f"\nSPLIT (train={n_train} val={n_val} test={n_test}):")
    seg_stats("train", train)
    seg_stats("val", val)
    seg_stats("test", test)

    print("\nRAW-UNITS naive persistence (no scaling, directly on each segment):")
    for h in (1, 48):
        mtr, rtr = persistence_r2(train, h)
        mte, rte = persistence_r2(test, h)
        print(f"  horizon={h:>2}: TRAIN MAE={mtr:6.2f} R2={rtr:7.4f} | "
              f"TEST MAE={mte:6.2f} R2={rte:7.4f}")

    # Lag-1 autocorrelation per segment.
    def acf1(a):
        a = a - a.mean()
        return float(np.sum(a[1:] * a[:-1]) / np.sum(a * a))
    print(f"\nLag-1 autocorrelation: train={acf1(train):.4f} test={acf1(test):.4f}")

    # How much does the test segment jump step-to-step?
    d = np.abs(np.diff(test))
    print(f"Test |step diff|: mean={d.mean():.2f} median={np.median(d):.2f} "
          f"max={d.max():.2f}  (>10% jumps: {(d > 10).sum()}/{len(d)})")

    # Compare ALL candidate files at horizon=1 on their OWN test tail.
    print("\nAll NAB files — test-tail 1-step persistence R2 (higher=more forecastable):")
    rows = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".csv"):
            continue
        sv = pd.read_csv(os.path.join(data_dir, fn))["value"].values.astype(float)
        nt = int(len(sv) * dl.TEST_RATIO)
        nv = int(len(sv) * dl.VAL_RATIO)
        tail = sv[len(sv) - nt:]
        whole = sv
        _, r2_tail = persistence_r2(tail, 1)
        _, r2_full = persistence_r2(whole, 1)
        rows.append((fn, sv.mean(), sv.std(), r2_full, r2_tail))
    rows.sort(key=lambda r: -r[4])
    print(f"  {'file':<34}{'mean':>7}{'std':>7}{'R2_full':>9}{'R2_tail':>9}")
    for fn, mu, sd, rf, rt in rows:
        print(f"  {fn:<34}{mu:7.2f}{sd:7.2f}{rf:9.4f}{rt:9.4f}")


if __name__ == "__main__":
    main()
