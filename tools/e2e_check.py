"""End-to-end check: load the REAL exported model via the inference app and
round-trip a real CPU window. Run as a file:

    PYTHONPATH=src .venv/Scripts/python.exe tools/e2e_check.py
"""
import os
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["MODEL_DIR"] = str(ROOT / "model_export")
os.environ["CLOUDSENSE_SRC"] = str(ROOT / "src")

spec = importlib.util.spec_from_file_location("cs_app", ROOT / "deploy" / "inference" / "app.py")
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)

from fastapi.testclient import TestClient  # noqa: E402

assert app_mod.STATE.ready, f"model not loaded: {app_mod.STATE.error}"
client = TestClient(app_mod.app)

print("=== /health ===")
print(json.dumps(client.get("/health").json(), indent=2))

print("\n=== /metrics ===")
print(json.dumps(client.get("/metrics").json(), indent=2))

# Real window: last look_back samples from the high-CPU test region of ac20cd.
import pandas as pd  # noqa: E402
cfg = app_mod.STATE.cfg
lb = cfg["look_back"]
csv = ROOT / "data" / "ec2_cpu_utilization_ac20cd.csv"
vals = pd.read_csv(csv)["value"].astype(float).values
window = vals[-(lb + 1):-1].tolist()          # last lb values before the final point
actual_next = float(vals[-1])                  # not the h-step target, just context

r = client.post("/predict", json={"cpu_percent": window})
print("\n=== /predict (real ac20cd tail) ===")
print(f"  status: {r.status_code}")
body = r.json()
print(json.dumps(body, indent=2))

pred = body["predicted_cpu_percent"]
assert 0.0 <= pred <= 100.0, "prediction out of range"
print(f"\n  window last value: {window[-1]:.2f}%   forecast (+{body['horizon_label']}): {pred:.2f}%")
print(f"  recommendation: {body['recommendation']}")
print("\nE2E OK")
