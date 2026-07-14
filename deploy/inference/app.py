"""
CloudSense inference API
========================
Serves the trained CEEMDAN+CNN-BiLSTM model exported by `src/main.py`.

Why this is faithful to the benchmark
-------------------------------------
The forecast here is produced by the SAME code paths used at training time:
the model classes come from `models_torch`, and the causal feature pipeline
(physical scaling -> RevIN per-window normalization -> causal CEEMDAN of the
supplied window) is `data_loader.features_from_raw_pct`. There is no separate,
approximate "inference CEEMDAN": the API decomposes only the past window you
send, exactly as the held-out benchmark did. So the live forecast reproduces
the reported test-set accuracy.

Run locally:
    PYTHONPATH=../../src uvicorn app:app --port 8000
    (or just `python app.py`)
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cloudsense.api")


# ---------------------------------------------------------------------------
# Make the project's src/ importable (shared model + feature code)
# ---------------------------------------------------------------------------
def _add_src_to_path() -> Optional[str]:
    candidates = [
        os.getenv("CLOUDSENSE_SRC"),
        str(Path(__file__).resolve().parents[2] / "src"),   # repo/src (local)
        "/app/src",                                          # Docker layout
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    return None


_SRC = _add_src_to_path()
try:
    import data_loader  # noqa: E402
    from models_torch import CEEMDANBiLSTM  # noqa: E402
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover - surfaced via /health
    data_loader = None
    CEEMDANBiLSTM = None
    _IMPORT_ERR = f"could not import src modules (CLOUDSENSE_SRC={_SRC!r}): {e}"
    log.error(_IMPORT_ERR)


# ---------------------------------------------------------------------------
# Load artifacts at startup (graceful: never crash the process)
# ---------------------------------------------------------------------------
def _default_model_dir() -> Path:
    env = os.getenv("MODEL_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "model_export"


MODEL_DIR = _default_model_dir()

# Autoscaling recommendation thresholds (CPU %), overridable via env.
SCALE_UP = float(os.getenv("CLOUDSENSE_SCALE_UP_PCT", "75"))
SCALE_DOWN = float(os.getenv("CLOUDSENSE_SCALE_DOWN_PCT", "30"))


class _State:
    model = None
    scaler = None
    cfg = None
    metrics = None
    ready = False
    error: Optional[str] = None


STATE = _State()


def load_artifacts() -> None:
    if _IMPORT_ERR:
        STATE.error = _IMPORT_ERR
        return
    try:
        with open(MODEL_DIR / "model_config.json") as f:
            cfg = json.load(f)
        model = CEEMDANBiLSTM(
            n_components=cfg["n_components"],
            look_back=cfg["look_back"],
            hidden=cfg["hidden"],
            conv_filters=cfg["conv_filters"],
        )
        state = torch.load(MODEL_DIR / "ceemdan_bilstm.pth", map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        metrics = None
        mpath = MODEL_DIR / "training_metrics.json"
        if mpath.exists():
            with open(mpath) as f:
                metrics = json.load(f)

        STATE.model, STATE.scaler, STATE.cfg, STATE.metrics = model, scaler, cfg, metrics
        STATE.ready = True
        STATE.error = None
        log.info("Loaded model from %s (horizon=%s, look_back=%s, n_components=%s)",
                 MODEL_DIR, cfg.get("horizon"), cfg.get("look_back"), cfg.get("n_components"))
    except FileNotFoundError as e:
        STATE.error = (f"model artifacts not found in {MODEL_DIR} ({e}). "
                       f"Run `python src/main.py` to train+export, or set MODEL_DIR.")
        log.error(STATE.error)
    except Exception as e:  # pragma: no cover
        STATE.error = f"failed to load artifacts: {e}"
        log.error(STATE.error)


load_artifacts()


# ---------------------------------------------------------------------------
# Core prediction (shared causal pipeline)
# ---------------------------------------------------------------------------
def _predict_pct(cpu_percent: List[float]) -> float:
    cfg = STATE.cfg
    look_back = cfg["look_back"]
    comps, mu, sd = data_loader.features_from_raw_pct(
        cpu_percent, look_back=look_back, n_components=cfg["n_components"],
        trials=cfg.get("trials", 20), seed=cfg.get("seed", 42), scaler=STATE.scaler)
    x_list = [torch.tensor(comps[k], dtype=torch.float32).reshape(1, look_back, 1)
              for k in range(cfg["n_components"])]
    with torch.no_grad():
        pred_norm = float(STATE.model(x_list).item())
    pred_scaled = min(max(pred_norm * sd + mu, 0.0), 1.0)        # undo RevIN + clip
    return float(STATE.scaler.inverse_transform([[pred_scaled]])[0][0])  # -> %


def _recommendation(pred_pct: float) -> str:
    if pred_pct >= SCALE_UP:
        return "scale_up"
    if pred_pct <= SCALE_DOWN:
        return "scale_down"
    return "hold"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CloudSense API",
    description="Causal CEEMDAN+CNN-BiLSTM cloud-workload forecaster.",
    version="3.0.0",
)

# CORS: explicit allow-list from env (NO wildcard default). Comma-separated.
_origins = [o.strip() for o in os.getenv("CLOUDSENSE_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])


class PredictRequest(BaseModel):
    """Recent CPU-utilization history (%). Must hold >= look_back samples;
    only the most recent look_back are used."""
    cpu_percent: List[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predicted_cpu_percent: float
    horizon_steps: int
    horizon_label: str
    look_back: int
    model: str
    decomposition: str
    recommendation: str
    thresholds: dict
    unit: str = "cpu_utilisation_percent"


def _require_ready():
    if not STATE.ready:
        raise HTTPException(status_code=503, detail=STATE.error or "model not loaded")


@app.get("/health")
def health():
    cfg = STATE.cfg or {}
    return {
        "status": "ok" if STATE.ready else "unavailable",
        "model_loaded": STATE.ready,
        "error": STATE.error,
        "model": cfg.get("architecture"),
        "look_back": cfg.get("look_back"),
        "horizon_steps": cfg.get("horizon"),
        "horizon_label": cfg.get("horizon_label"),
        "n_components": cfg.get("n_components"),
        "decomposition": cfg.get("decomposition"),
        "causal_windowed_decomposition": cfg.get("causal_windowed_decomposition"),
    }


@app.get("/metrics")
def metrics():
    """Held-out test-set scores recorded at export time (honest, reproducible)."""
    _require_ready()
    return STATE.metrics or {"note": "training_metrics.json was not exported"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _require_ready()
    cfg = STATE.cfg
    look_back = cfg["look_back"]
    vals = req.cpu_percent
    if len(vals) < look_back:
        raise HTTPException(
            status_code=422,
            detail=f"cpu_percent needs >= {look_back} samples, got {len(vals)}")
    if not all(np.isfinite(vals)):
        raise HTTPException(status_code=422, detail="cpu_percent contains non-finite values")
    if min(vals) < -0.001 or max(vals) > 100.001:
        raise HTTPException(status_code=422, detail="cpu_percent values must be within [0, 100]")

    pred = _predict_pct(vals)
    return PredictResponse(
        predicted_cpu_percent=round(pred, 3),
        horizon_steps=cfg["horizon"],
        horizon_label=cfg.get("horizon_label", str(cfg["horizon"])),
        look_back=look_back,
        model=cfg.get("architecture", "CEEMDAN+CNN-BiLSTM"),
        decomposition=f"{cfg.get('decomposition', 'CEEMDAN')} (causal, per-window)",
        recommendation=_recommendation(pred),
        thresholds={"scale_up_pct": SCALE_UP, "scale_down_pct": SCALE_DOWN},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
