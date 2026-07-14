"""Live smoke test for a RUNNING CloudSense API (local or deployed).

    python deploy/test/test_api.py                       # hits 127.0.0.1:8000
    python deploy/test/test_api.py --url http://EC2_IP:8000

For the in-process unit test (no running server needed) see tests/test_api.py.
"""
import argparse
import sys
import random

import requests


def test_health(base):
    print("\n[1] Health ...")
    d = requests.get(f"{base}/health", timeout=10).json()
    assert d.get("model_loaded"), f"model not loaded: {d.get('error')}"
    print(f"    [OK] look_back={d['look_back']} horizon={d.get('horizon_label')} "
          f"decomposition={d.get('decomposition')}")
    return d["look_back"]


def test_metrics(base):
    print("\n[2] Test-set metrics ...")
    d = requests.get(f"{base}/metrics", timeout=10).json()
    m = d.get("metrics", {})
    print(f"    MAE={m.get('MAE')}  RMSE={m.get('RMSE')}  R2={m.get('R2')}")


def test_predict(base, look_back):
    print("\n[3] Forecast (steady ~55%) ...")
    seq = [55 + random.gauss(0, 5) for _ in range(look_back)]
    d = requests.post(f"{base}/predict", json={"cpu_percent": seq}, timeout=30).json()
    print(f"    predicted={d['predicted_cpu_percent']:.2f}%  rec={d['recommendation']}")

    print("\n[4] Forecast (high ~85% -> expect scale_up) ...")
    seq_hi = [85 + random.gauss(0, 3) for _ in range(look_back)]
    d2 = requests.post(f"{base}/predict", json={"cpu_percent": seq_hi}, timeout=30).json()
    print(f"    predicted={d2['predicted_cpu_percent']:.2f}%  rec={d2['recommendation']}")


def test_validation(base):
    print("\n[5] Validation (too few points -> 422) ...")
    r = requests.post(f"{base}/predict", json={"cpu_percent": [50.0] * 10}, timeout=10)
    print("    [OK] 422" if r.status_code == 422 else f"    [FAIL] got {r.status_code}")


def main():
    ap = argparse.ArgumentParser(description="CloudSense live API smoke test")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    ap.add_argument("--ip", help="shortcut: host -> http://<ip>:8000")
    args = ap.parse_args()
    base = f"http://{args.ip}:8000" if args.ip else args.url.rstrip("/")

    print(f"{'=' * 52}\n  CloudSense API smoke test -> {base}\n{'=' * 52}")
    try:
        look_back = test_health(base)
        test_metrics(base)
        test_predict(base, look_back)
        test_validation(base)
    except requests.exceptions.ConnectionError:
        print(f"\n[FAIL] cannot connect to {base}. Is the API running?")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    print(f"\n{'=' * 52}\n  [OK] done\n{'=' * 52}")


if __name__ == "__main__":
    main()
