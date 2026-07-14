"""Contract tests for deploy/inference/app.py.

Builds a tiny dummy model_export (random weights, small architecture, few CEEMDAN
trials so it's fast), boots the FastAPI app against it with a TestClient, and
checks the real causal predict path + validation. No network, no trained model
needed -- this verifies the API CONTRACT, not accuracy.
"""
import json
import os
import pickle
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "deploy" / "inference" / "app.py"

LOOK_BACK = 48
N_COMPONENTS = 4          # small for speed
HIDDEN = 8
CONV = 4
TRIALS = 3                # fast causal CEEMDAN on a 48-pt window


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    import models_torch as mt
    from sklearn.preprocessing import MinMaxScaler
    import torch

    export = tmp_path_factory.mktemp("model_export")

    model = mt.CEEMDANBiLSTM(n_components=N_COMPONENTS, look_back=LOOK_BACK,
                             hidden=HIDDEN, conv_filters=CONV)
    torch.save(model.state_dict(), export / "ceemdan_bilstm.pth")

    cfg = {
        "architecture": "CEEMDAN+CNN-BiLSTM", "look_back": LOOK_BACK,
        "horizon": 48, "horizon_label": "4h", "n_components": N_COMPONENTS,
        "hidden": HIDDEN, "conv_filters": CONV, "trials": TRIALS, "seed": 42,
        "decomposition": "CEEMDAN", "causal_windowed_decomposition": True,
        "normalization": "revin_window_meanstd", "eps_scaled": 0.01,
        "input_unit": "cpu_utilisation_percent", "input_range": [0.0, 100.0],
    }
    (export / "model_config.json").write_text(json.dumps(cfg))

    scaler = MinMaxScaler().fit(np.array([[0.0], [100.0]]))
    with open(export / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (export / "training_metrics.json").write_text(json.dumps(
        {"model": "CEEMDAN+CNN-BiLSTM", "metrics": {"MAE": 1.0, "R2": 0.9}}))

    # Import the app fresh with MODEL_DIR pointing at the dummy export.
    os.environ["MODEL_DIR"] = str(export)
    os.environ["CLOUDSENSE_SRC"] = str(ROOT / "src")
    spec = importlib.util.spec_from_file_location("cloudsense_app_under_test", APP_PATH)
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    assert app_mod.STATE.ready, f"app failed to load dummy model: {app_mod.STATE.error}"
    return fastapi_testclient.TestClient(app_mod.app)


def test_health_reports_ready_and_config(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["look_back"] == LOOK_BACK
    assert body["horizon_steps"] == 48
    assert body["causal_windowed_decomposition"] is True


def test_predict_valid_window(client):
    rng = np.random.default_rng(0)
    window = (40 + 10 * np.sin(np.linspace(0, 6, LOOK_BACK))
              + rng.normal(0, 2, LOOK_BACK)).clip(0, 100).tolist()
    r = client.post("/predict", json={"cpu_percent": window})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["predicted_cpu_percent"] <= 100.0
    assert body["horizon_steps"] == 48
    assert body["recommendation"] in {"scale_up", "hold", "scale_down"}
    assert body["look_back"] == LOOK_BACK


def test_predict_uses_only_last_look_back(client):
    # Sending more than look_back is allowed; only the most recent window is used.
    window = (np.full(LOOK_BACK + 30, 50.0)).tolist()
    r = client.post("/predict", json={"cpu_percent": window})
    assert r.status_code == 200, r.text


def test_predict_rejects_short_window(client):
    r = client.post("/predict", json={"cpu_percent": [50.0] * 10})
    assert r.status_code == 422


def test_predict_rejects_out_of_range(client):
    window = [50.0] * (LOOK_BACK - 1) + [150.0]
    r = client.post("/predict", json={"cpu_percent": window})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "metrics" in r.json()
