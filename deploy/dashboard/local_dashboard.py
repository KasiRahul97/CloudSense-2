"""
CloudSense demo dashboard (local).
==================================
A thin proxy + static UI in front of the inference API (deploy/inference/app.py).

* Forecast/health/metrics calls are PROXIED to the real model API
  (CLOUDSENSE_INFERENCE_URL, default http://127.0.0.1:8000).
* The "fleet" is an explicitly SIMULATED autoscaling group used to visualise how
  a scaler would react to the model's forecast. It does NOT touch real AWS.

Run:
    # 1) start the model API
    PYTHONPATH=../../src uvicorn app:app --port 8000          # in deploy/inference
    # 2) start this dashboard
    uvicorn local_dashboard:app --port 8501                   # in deploy/dashboard
"""
import os
import uuid
import asyncio
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="CloudSense dashboard")

# Real model API (proxied). No hardcoded IPs: configure via env.
INFERENCE_URL = os.getenv("CLOUDSENSE_INFERENCE_URL", "http://127.0.0.1:8000").rstrip("/")

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# SIMULATED autoscaling fleet (local demo only -- not real AWS)
# ---------------------------------------------------------------------------
sim_fleet = {
    "i-main-001": {"id": "i-main-001", "name": "CloudSense-Primary",
                   "status": "Running", "type": "t3.micro (simulated)", "is_main": True},
}


async def _simulate_boot(instance_id: str):
    await asyncio.sleep(4)
    if instance_id in sim_fleet:
        sim_fleet[instance_id]["status"] = "Running"


@app.get("/api/fleet")
def get_fleet():
    return sim_fleet


@app.post("/api/fleet/provision")
async def provision_instance():
    new_id = "i-" + str(uuid.uuid4())[:8]
    sim_fleet[new_id] = {"id": new_id, "name": "CloudSense-Worker-" + str(uuid.uuid4())[:3],
                         "status": "Pending", "type": "t3.micro (simulated)", "is_main": False}
    asyncio.create_task(_simulate_boot(new_id))
    return {"status": "provisioning", "id": new_id}


@app.post("/api/fleet/{instance_id}/stop")
def stop_instance(instance_id: str):
    if instance_id in sim_fleet:
        sim_fleet[instance_id]["status"] = "Stopped"
    return {"status": "ok"}


@app.post("/api/fleet/{instance_id}/start")
async def start_instance(instance_id: str):
    if instance_id in sim_fleet:
        sim_fleet[instance_id]["status"] = "Pending"
        asyncio.create_task(_simulate_boot(instance_id))
    return {"status": "ok"}


@app.post("/api/fleet/{instance_id}/terminate")
def terminate_instance(instance_id: str):
    if instance_id in sim_fleet and not sim_fleet[instance_id].get("is_main"):
        del sim_fleet[instance_id]
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Proxy to the real model API
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    cpu_percent: list[float]


def _proxy_get(path: str):
    try:
        r = requests.get(f"{INFERENCE_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"inference API unreachable: {e}")


@app.get("/api/health")
def get_health():
    return _proxy_get("/health")


@app.get("/api/metrics")
def get_metrics():
    return _proxy_get("/metrics")


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        r = requests.post(f"{INFERENCE_URL}/predict",
                          json={"cpu_percent": req.cpu_percent}, timeout=20)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"inference API unreachable: {e}")


@app.get("/api/sample")
def get_sample(n: int = 96):
    """Return a real recent CPU window from the cached NAB dataset (so the demo
    runs on genuine data), falling back to a synthetic wave if it is absent."""
    import numpy as np
    try:
        import pandas as pd
        csv = ROOT / "data" / "ec2_cpu_utilization_ac20cd.csv"
        vals = pd.read_csv(csv)["value"].astype(float).values
        # A dynamic stretch from the high-utilisation test region.
        seg = vals[int(len(vals) * 0.85): int(len(vals) * 0.85) + n]
        if len(seg) >= 8:
            return {"cpu_percent": [round(float(v), 2) for v in seg], "source": "numenta_nab"}
    except Exception:
        pass
    t = np.linspace(0, 6, n)
    seg = np.clip(45 + 20 * np.sin(t) + np.random.normal(0, 3, n), 0, 100)
    return {"cpu_percent": [round(float(v), 2) for v in seg], "source": "synthetic"}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
