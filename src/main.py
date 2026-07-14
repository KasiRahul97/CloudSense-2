"""
main.py
=======
End-to-end CloudSense benchmark.

Runs the SAME comparison at every forecast horizon in HORIZONS:

    Persistence  EMA            <- naive sanity baselines (no training)
    LSTM  CNN-LSTM  Bi-LSTM  Transformer   <- peer-reviewed learned baselines
    CEEMDAN + CNN-BiLSTM ensemble           <- proposed model

Every model at a given horizon is scored against the one shared ground-truth
array (train_evaluate.test_targets), so the numbers are directly comparable.
horizon=1 is a 5-minute-ahead forecast; horizon=48 is 4 hours ahead.

Outputs (project root):
    results/metrics.csv      one row per (horizon, model)
    results/summary.txt      human-readable table, grouped by horizon
    figures/*.png            raw data, decomposition, per-horizon predictions
                             and per-horizon metric bars
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_dataset, ceemdan_available
from models_torch import (LSTMModel, CNNLSTMModel, BiLSTMModel,
                          TransformerModel, CEEMDANBiLSTM)
from train_evaluate import (train_model, evaluate_model,
                            train_ceemdan_model, evaluate_ceemdan_model,
                            evaluate_persistence, evaluate_ema, DEVICE)

# ---------------------------------------------------------------------------
# Paths (anchored to the project root, not the current working directory)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.getenv("CLOUDSENSE_DATA_DIR", os.path.join(ROOT, "data"))
FIG_DIR = os.getenv("CLOUDSENSE_FIG_DIR", os.path.join(ROOT, "figures"))
RES_DIR = os.getenv("CLOUDSENSE_RES_DIR", os.path.join(ROOT, "results"))
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default

# Knobs are env-overridable so the same script serves a fast smoke test and the
# full benchmark, e.g. CLOUDSENSE_HORIZONS=1 CLOUDSENSE_EPOCHS=3 CLOUDSENSE_TRIALS=3.
LOOK_BACK = 48        # 4-hour input window (48 * 5 min)
N_COMPONENTS = _env_int("CLOUDSENSE_COMPONENTS", 8)   # == number of ensemble sub-models
HORIZONS = [int(h) for h in os.getenv("CLOUDSENSE_HORIZONS", "1,48").split(",") if h.strip()]
EPOCHS = _env_int("CLOUDSENSE_EPOCHS", 120)
LR = 1e-3
BATCH = 64
PATIENCE = _env_int("CLOUDSENSE_PATIENCE", 15)
SEED = _env_int("CLOUDSENSE_SEED", 42)
TRIALS = _env_int("CLOUDSENSE_TRIALS", 20)            # CEEMDAN noise-ensemble trials

# Proposed-model capacity (kept small: one sub-model per component, so the
# ensemble already has N_COMPONENTS of these). MUST match what the inference
# service rebuilds from model_export/model_config.json.
PROP_HIDDEN = 32
PROP_CONV = 16

# The trained proposed model is exported once so the FastAPI inference service
# in deploy/ can load and serve it. We export the LONGEST horizon (4 h by
# default): a multi-hour-ahead forecast is what actually enables proactive
# autoscaling, which is the project's headline use case.
EXPORT_DIR = os.getenv("CLOUDSENSE_EXPORT_DIR", os.path.join(ROOT, "model_export"))
EXPORT_HORIZON = HORIZONS[-1]

PROPOSED = "CEEMDAN+CNN-BiLSTM (Proposed)"

MODEL_COLORS = {
    "Persistence": "#adb5bd",
    "EMA (a=0.3)": "#868e96",
    "LSTM": "#e76f51",
    "CNN-LSTM": "#2a9d8f",
    "Bi-LSTM": "#e9c46a",
    "Transformer": "#457b9d",
    PROPOSED: "#d62828",
    "Actual": "#264653",
}


def horizon_label(h):
    minutes = h * 5
    return f"{minutes // 60}h" if minutes >= 60 else f"{minutes}min"


def build_nn_models():
    """Fresh learned-baseline instances (one set per horizon)."""
    return [
        LSTMModel(hidden_size=64, num_layers=2, dropout=0.2, look_back=LOOK_BACK),
        CNNLSTMModel(cnn_filters=32, hidden_size=64, num_layers=2,
                     dropout=0.2, look_back=LOOK_BACK),
        BiLSTMModel(hidden_size=64, num_layers=2, dropout=0.2, look_back=LOOK_BACK),
        TransformerModel(d_model=64, nhead=4, num_layers=2,
                         dim_feedforward=128, dropout=0.1, look_back=LOOK_BACK),
    ]


# ---------------------------------------------------------------------------
# Model export (so deploy/inference can serve the trained proposed model)
# ---------------------------------------------------------------------------
def export_artifacts(model, data, metrics, out_dir):
    """Persist everything the inference service needs to reproduce a forecast:

        ceemdan_bilstm.pth     trained ensemble weights (state_dict)
        model_config.json      hyper-params to rebuild the architecture + the
                               causal feature pipeline the API must reuse
        scaler.pkl             physical [0,100]% MinMaxScaler (raw% <-> [0,1])
        training_metrics.json  honest held-out test scores for this model

    The config/scaler/weights triple is a hard contract with deploy/inference:
    the API rebuilds CEEMDANBiLSTM from model_config.json and reuses
    data_loader.features_from_raw_pct with these exact settings, so the live
    forecast reproduces the benchmark. n_components/look_back/hidden/conv_filters
    MUST match run_horizon; normalization/decomposition MUST match data_loader.
    """
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "ceemdan_bilstm.pth"))

    config = {
        "architecture": "CEEMDAN+CNN-BiLSTM",
        "look_back": LOOK_BACK,
        "horizon": data["horizon"],
        "horizon_label": horizon_label(data["horizon"]),
        "n_components": N_COMPONENTS,
        "hidden": PROP_HIDDEN,
        "conv_filters": PROP_CONV,
        "trials": TRIALS,
        "seed": SEED,
        "decomposition": data.get("decomposition",
                                  "CEEMDAN" if ceemdan_available() else "moving_average_fallback"),
        "causal_windowed_decomposition": True,
        "normalization": data.get("normalization", "revin_window_meanstd"),
        "eps_scaled": data.get("eps_scaled", 1e-2),
        "input_unit": "cpu_utilisation_percent",
        "input_range": [0.0, 100.0],
    }
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(data["scaler"], f)

    metrics_out = {
        "dataset": "synthetic_fallback" if data["is_synthetic"] else "numenta_nab_ec2_cpu",
        "model": PROPOSED,
        "horizon": data["horizon"],
        "horizon_label": horizon_label(data["horizon"]),
        "metrics": {k: metrics.get(k) for k in ("MAE", "RMSE", "MAPE", "R2")},
        "unit": "cpu_utilisation_percent",
        "decomposition": config["decomposition"],
        "note": "Held-out test-set scores. Reproduce with src/main.py (seed=%d)." % SEED,
    }
    with open(os.path.join(out_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"  [export] proposed model + scaler + config -> "
          f"{os.path.relpath(out_dir, ROOT)}/")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_horizon(h):
    """Train + evaluate every model at horizon h. Returns results dict."""
    print(f"\n[ horizon {h} ({horizon_label(h)} ahead) ] loading + decomposing ...")
    data = load_dataset(data_dir=DATA_DIR, look_back=LOOK_BACK, horizon=h,
                        n_components=N_COMPONENTS, trials=TRIALS, seed=SEED)
    results = {}

    def record(name, triple, hist=None):
        preds, y_true, metrics = triple
        results[name] = {"preds": preds, "y_true": y_true,
                         "metrics": metrics, "hist": hist}
        print(f"    {name:<32} "
              f"MAE={metrics.get('MAE', float('nan')):>7.3f} "
              f"RMSE={metrics.get('RMSE', float('nan')):>7.3f} "
              f"MAPE={metrics.get('MAPE', float('nan')):>6.2f}% "
              f"R2={metrics.get('R2', float('nan')):>7.4f}")

    # Naive baselines -- no training.
    record("Persistence", evaluate_persistence(data))
    record("EMA (a=0.3)", evaluate_ema(data, alpha=0.3))

    # Learned baselines.
    for model in build_nn_models():
        print(f"  training {model.name} ...")
        model, hist = train_model(model, data, epochs=EPOCHS, lr=LR,
                                  batch_size=BATCH, patience=PATIENCE,
                                  seed=SEED, verbose=False)
        record(model.name, evaluate_model(model, data), hist)

    # Proposed ensemble.
    print(f"  training {PROPOSED} ({N_COMPONENTS} sub-models) ...")
    proposed = CEEMDANBiLSTM(n_components=N_COMPONENTS, look_back=LOOK_BACK,
                             hidden=PROP_HIDDEN, conv_filters=PROP_CONV)
    proposed, phist = train_ceemdan_model(proposed, data, epochs=EPOCHS, lr=LR,
                                          batch_size=BATCH, patience=PATIENCE,
                                          seed=SEED, verbose=False)
    record(PROPOSED, evaluate_ceemdan_model(proposed, data), phist)

    # Export the trained proposed model once, so deploy/inference can serve it.
    if h == EXPORT_HORIZON:
        export_artifacts(proposed, data, results[PROPOSED]["metrics"], EXPORT_DIR)

    return data, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def save_metrics(all_rows):
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RES_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")

    txt_path = os.path.join(RES_DIR, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("  CloudSense -- comparative forecasting results\n")
        f.write(f"  Decomposition: {'CEEMDAN (PyEMD)' if ceemdan_available() else 'moving-average FALLBACK'}\n")
        f.write(f"  Device: {DEVICE}\n")
        f.write("=" * 72 + "\n")
        for h in sorted(df["Horizon"].unique()):
            sub = df[df["Horizon"] == h]
            f.write(f"\nHorizon {h} ({horizon_label(h)} ahead)\n")
            f.write(f"{'Model':<34}{'MAE':>9}{'RMSE':>9}{'MAPE%':>9}{'R2':>9}\n")
            f.write("-" * 70 + "\n")
            for _, r in sub.iterrows():
                star = "  *" if r["Model"] == PROPOSED else ""
                f.write(f"{r['Model']:<34}{r['MAE']:>9.3f}{r['RMSE']:>9.3f}"
                        f"{r['MAPE']:>9.2f}{r['R2']:>9.4f}{star}\n")
        f.write("\n* = proposed model\n")
    print(f"  -> {txt_path}")


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {os.path.relpath(path, ROOT)}")


def fig_raw_data(data):
    df = data["raw_df"]
    n = min(2016, len(df))
    y = df["cpu_util"].values[:n]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(y, color=MODEL_COLORS["Actual"], lw=0.7)
    ax.fill_between(range(n), y, alpha=0.10, color=MODEL_COLORS["Actual"])
    tag = " (SYNTHETIC fallback)" if data["is_synthetic"] else " (Numenta NAB)"
    ax.set_title(f"AWS EC2 CPU utilisation{tag} -- 7-day snapshot", fontweight="bold")
    ax.set_xlabel("Time step (5-min intervals)")
    ax.set_ylabel("CPU utilisation (%)")
    savefig("fig1_raw_data.png")


def fig_decomposition(data):
    """Show the ACTUAL causal, RevIN-normalized decomposition of one test window
    -- i.e. exactly the per-component input the proposed model (and the deployed
    API) receives. comps sum to the normalized window."""
    test = data["splits"]["test"]
    if len(test["Xc"]) == 0:
        return
    k = len(test["Xc"]) // 2                       # a representative window
    comps = test["Xc"][k]                          # (n_components, look_back)
    window = comps.sum(axis=0)                     # == RevIN-normalized window
    method = "CEEMDAN" if ceemdan_available() else "moving-average (fallback)"
    n_comp = comps.shape[0]
    fig, axes = plt.subplots(n_comp + 1, 1,
                             figsize=(13, 1.7 * (n_comp + 1)), sharex=True)
    fig.suptitle(f"Causal {method} decomposition of one RevIN-normalized "
                 f"{LOOK_BACK}-step input window (what the model sees)",
                 fontweight="bold")
    axes[0].plot(window, color=MODEL_COLORS["Actual"], lw=0.9)
    axes[0].set_ylabel("window", fontsize=8)
    for i, (ax, comp) in enumerate(zip(axes[1:], comps)):
        lbl = "residue" if i == n_comp - 1 else f"IMF {i + 1}"
        ax.plot(comp, lw=0.9)
        ax.axhline(0, color="#ccc", lw=0.5, ls="--")
        ax.set_ylabel(lbl, fontsize=8)
    axes[-1].set_xlabel("Step within the 4-hour window (5-min intervals)")
    plt.tight_layout()
    savefig("fig2_decomposition.png")


def fig_predictions(h, results):
    ref = results["Persistence"]["y_true"]
    n_show = min(576, len(ref))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ref[:n_show], color=MODEL_COLORS["Actual"], lw=1.8,
            label="Actual", zorder=10)
    for name, r in results.items():
        preds = r["preds"]
        if len(preds) == 0:
            continue
        is_prop = name == PROPOSED
        m = min(n_show, len(preds))
        ax.plot(preds[:m], color=MODEL_COLORS.get(name, "#888"),
                lw=1.8 if is_prop else 0.9, ls="-" if is_prop else "--",
                alpha=1.0 if is_prop else 0.6, label=name,
                zorder=11 if is_prop else 8)
    ax.set_title(f"Actual vs predicted CPU -- {horizon_label(h)} ahead "
                 f"(2-day test window)", fontweight="bold")
    ax.set_xlabel("Time step (5-min intervals)")
    ax.set_ylabel("CPU (%)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    savefig(f"fig3_predictions_h{h}.png")


def fig_metric_bars(h, results):
    names = list(results.keys())
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    titles = {"MAE": "MAE (lower better)", "RMSE": "RMSE (lower better)",
              "MAPE": "MAPE % (lower better)", "R2": "R2 (higher better)"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Metrics at horizon {h} ({horizon_label(h)} ahead)",
                 fontweight="bold")
    for ax, metric in zip(axes.ravel(), metrics):
        vals = [results[n]["metrics"].get(metric, np.nan) for n in names]
        colors = [MODEL_COLORS.get(n, "#888") for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
        if PROPOSED in names:
            bars[names.index(PROPOSED)].set_edgecolor("#d62828")
            bars[names.index(PROPOSED)].set_linewidth(2.5)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(titles[metric])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(" (")[0] for n in names],
                           rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    savefig(f"fig4_metrics_h{h}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("  CloudSense | Cloud resource usage forecasting benchmark")
    print(f"  Device: {DEVICE}")
    print(f"  Decomposition: "
          f"{'CEEMDAN (PyEMD)' if ceemdan_available() else 'moving-average FALLBACK (PyEMD missing)'}")
    print(f"  Horizons: {HORIZONS}  |  components: {N_COMPONENTS}  |  seed: {SEED}")
    print("=" * 64)

    all_rows = []
    first_data = None
    for h in HORIZONS:
        data, results = run_horizon(h)
        if first_data is None:
            first_data = data
        for name, r in results.items():
            all_rows.append({"Horizon": h, "HorizonLabel": horizon_label(h),
                             "Model": name, **r["metrics"]})
        fig_predictions(h, results)
        fig_metric_bars(h, results)

    save_metrics(all_rows)
    fig_raw_data(first_data)
    fig_decomposition(first_data)

    print("\n" + "=" * 64)
    print("  DONE -> results/  and  figures/")
    print("=" * 64)


if __name__ == "__main__":
    main()
