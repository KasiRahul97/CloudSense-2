"""Multi-seed robustness check at the 4-hour horizon.

Runs the full benchmark for several seeds (each into its own temp output dir so
the canonical results/ is untouched), then reports per-model mean +/- std of R2
and MAE. Strengthens the headline finding (is "naive baselines win" stable?).

    PYTHONPATH=src .venv/Scripts/python.exe tools/robustness.py
"""
import os
import sys
import csv
import subprocess
import tempfile
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [int(s) for s in os.getenv("CLOUDSENSE_ROBUST_SEEDS", "42,7,123").split(",")]
PY = sys.executable


def run_seed(seed: int, outdir: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "CLOUDSENSE_HORIZONS": "48",
        "CLOUDSENSE_SEED": str(seed),
        "CLOUDSENSE_EPOCHS": "120",
        "CLOUDSENSE_TRIALS": "20",
        "CLOUDSENSE_PATIENCE": "15",
        "CLOUDSENSE_RES_DIR": str(outdir / "results"),
        "CLOUDSENSE_FIG_DIR": str(outdir / "figures"),
        "CLOUDSENSE_EXPORT_DIR": str(outdir / "model_export"),
        "PYTHONPATH": str(ROOT / "src"),
    })
    print(f"[seed {seed}] running ...", flush=True)
    r = subprocess.run([PY, str(ROOT / "src" / "main.py")], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"seed {seed} failed (exit {r.returncode})")
    rows = {}
    with open(outdir / "results" / "metrics.csv") as f:
        for row in csv.DictReader(f):
            if int(row["Horizon"]) == 48:
                rows[row["Model"]] = {"R2": float(row["R2"]), "MAE": float(row["MAE"])}
    return rows


def main():
    per_seed = {}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            per_seed[seed] = run_seed(seed, Path(tmp) / f"seed_{seed}")

    models = list(per_seed[SEEDS[0]].keys())
    lines = ["# Multi-seed robustness (4-hour horizon, h=48)", "",
             f"Seeds: {SEEDS}. Each value is mean +/- sample std across seeds.", "",
             "| Model | R2 (mean +/- std) | MAE (mean +/- std) |",
             "|---|---|---|"]
    summary = []
    for m in models:
        r2s = [per_seed[s][m]["R2"] for s in SEEDS]
        maes = [per_seed[s][m]["MAE"] for s in SEEDS]
        r2_sd = statistics.stdev(r2s) if len(r2s) > 1 else 0.0
        mae_sd = statistics.stdev(maes) if len(maes) > 1 else 0.0
        summary.append((m, statistics.mean(r2s), r2_sd, statistics.mean(maes), mae_sd))
    summary.sort(key=lambda x: -x[1])
    for m, r2m, r2s, maem, maes in summary:
        lines.append(f"| {m} | {r2m:.3f} +/- {r2s:.3f} | {maem:.2f} +/- {maes:.2f} |")

    out = ROOT / "results" / "robustness_h48.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
