"""Live two-server end-to-end test: boot the real API + the dashboard as actual
uvicorn servers, drive a forecast through the dashboard proxy over real HTTP,
then shut both down. Owns the processes via subprocess so nothing is orphaned.

    PYTHONPATH=src .venv/Scripts/python.exe tools/live_e2e.py
"""
import os
import sys
import time
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
API_PORT, DASH_PORT = 8077, 8078


def wait_ready(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    api_env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                   MODEL_DIR=str(ROOT / "model_export"),
                   CLOUDSENSE_SRC=str(ROOT / "src"))
    dash_env = dict(os.environ,
                    CLOUDSENSE_INFERENCE_URL=f"http://127.0.0.1:{API_PORT}")

    api = subprocess.Popen([PY, "-m", "uvicorn", "app:app", "--port", str(API_PORT)],
                           cwd=str(ROOT / "deploy" / "inference"), env=api_env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dash = subprocess.Popen([PY, "-m", "uvicorn", "local_dashboard:app", "--port", str(DASH_PORT)],
                            cwd=str(ROOT / "deploy" / "dashboard"), env=dash_env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        print("waiting for API ...")
        assert wait_ready(f"http://127.0.0.1:{API_PORT}/health"), "API did not start"
        print("waiting for dashboard ...")
        assert wait_ready(f"http://127.0.0.1:{DASH_PORT}/api/health"), "dashboard did not start"

        base = f"http://127.0.0.1:{DASH_PORT}"
        health = requests.get(f"{base}/api/health").json()
        print(f"  dashboard -> API /health: model_loaded={health['model_loaded']} "
              f"horizon={health.get('horizon_label')} model={health.get('model')}")

        sample = requests.get(f"{base}/api/sample").json()
        print(f"  /api/sample: {len(sample['cpu_percent'])} pts from {sample['source']}")

        pred = requests.post(f"{base}/api/predict",
                             json={"cpu_percent": sample["cpu_percent"]}).json()
        print(f"  /api/predict -> forecast {pred['predicted_cpu_percent']:.2f}% "
              f"(+{pred['horizon_label']}), rec={pred['recommendation']}")

        assert 0 <= pred["predicted_cpu_percent"] <= 100
        print("\nLIVE 2-SERVER E2E OK (browser -> dashboard -> API -> model)")
    finally:
        for p in (dash, api):
            p.terminate()
        for p in (dash, api):
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        print("servers shut down.")


if __name__ == "__main__":
    main()
