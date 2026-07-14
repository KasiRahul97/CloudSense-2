# CloudSense

**A reproducible, leakage-free benchmark of CPU-utilization forecasters for
proactive cloud autoscaling — plus a deployed model-serving stack whose live
forecasts reproduce the held-out test accuracy.**

CloudSense forecasts AWS EC2 CPU utilization several hours ahead so an
autoscaler can act *before* load arrives. It compares a CEEMDAN + CNN-BiLSTM
ensemble against four standard deep-learning baselines and two naive baselines
on real [Numenta NAB](https://github.com/numenta/NAB) data, then serves the
trained model behind a FastAPI service with a live dashboard.

> **Honest headline:** under a fair, causal, leakage-free evaluation, **naive
> persistence is a very strong baseline** and is not beaten by any learned model
> on this single-instance series. The proposed CEEMDAN+CNN-BiLSTM is *competitive
> among the learned models* but is not the overall winner. The value of this
> project is the **rigor of the evaluation and the working end-to-end system**,
> not a state-of-the-art claim. See [Results](#results).

---

## Why this project is interesting (the engineering, not a SOTA claim)

- **It is causal and leakage-free.** CEEMDAN (an EMD-family decomposition) is
  non-causal — a component's value at time *t* normally depends on the whole
  signal, including the future. CloudSense decomposes **only the past input
  window** (`data_loader.window_components`), the *same* code the API runs at
  serving time. So the benchmark number is something the deployed model can
  actually reproduce online, not a number that only exists offline.
- **It handles non-stationarity honestly.** The chosen series has a real
  train/test regime shift (train averages ~33% CPU, the test tail averages
  ~71%). Per-window **RevIN normalization** makes every model level-invariant so
  it can transfer across that shift. Without it, the learned baselines collapse
  to negative R² (e.g. LSTM at the 4-hour horizon went from **R² ≈ −1.70**
  without RevIN to **≈ 0.70** with it).
- **It includes the baselines that matter.** Persistence and EMA are reported
  alongside the deep models. A learned model that cannot beat "tomorrow looks
  like today" is not useful, and this benchmark says so out loud.
- **It is a real system, not a notebook.** Train → export artifacts → FastAPI
  inference (Docker) → dashboard. The API loads the exact `src/` code used in
  training, so serving == benchmark.

---

## Results

Real Numenta NAB EC2 series `ec2_cpu_utilization_ac20cd` (4032 points, 5-min
cadence). Single instance, temporal 70/10/20 train/val/test split, seed 42.
All metrics are on the held-out **test** set in real CPU-% units. Lower
MAE/RMSE/MAPE is better; higher R² is better. The shared ground truth is
identical across models.

### Horizon = 1 step (5 minutes ahead)
| Model | MAE | RMSE | MAPE% | R² |
|---|---:|---:|---:|---:|
| **Persistence** (naive) | 1.38 | 3.08 | 3.38 | **0.991** |
| **EMA** (naive) | **1.23** | 3.54 | 2.66 | 0.988 |
| CNN-LSTM | 1.72 | 5.49 | 2.93 | 0.971 |
| Bi-LSTM | 1.41 | 5.60 | 2.64 | 0.970 |
| LSTM | 1.78 | 5.73 | 2.98 | 0.968 |
| CEEMDAN+CNN-BiLSTM (proposed) | 1.81 | 6.47 | 3.06 | 0.960 |
| Transformer | 2.40 | 8.13 | 3.56 | 0.936 |

### Horizon = 48 steps (4 hours ahead) — the operationally useful horizon
| Model | MAE | RMSE | MAPE% | R² |
|---|---:|---:|---:|---:|
| **Persistence** (naive) | 4.95 | 15.91 | 6.52 | **0.756** |
| Transformer | 5.68 | 15.99 | 7.15 | 0.754 |
| **EMA** (naive) | **4.90** | 16.04 | 6.14 | 0.753 |
| CEEMDAN+CNN-BiLSTM (proposed) | 5.72 | 16.75 | 7.48 | 0.730 |
| Bi-LSTM | 6.14 | 17.44 | 7.45 | 0.708 |
| LSTM | 6.27 | 17.59 | 7.65 | 0.702 |
| CNN-LSTM | 8.91 | 21.27 | 13.84 | 0.565 |

**What I take from this**

1. CPU utilization on this instance is highly autocorrelated, so **persistence is
   hard to beat** even 4 hours out (R² ≈ 0.76). This is the single most important
   sanity check and most workload-forecasting write-ups omit it.
2. **RevIN normalization is what makes the learned models viable** at all under
   the regime shift; it is the highest-leverage idea in the pipeline.
3. Among learned models the proposed ensemble is **competitive but not best**
   (the Transformer edges it at 4 h). The decomposition front-end did not buy a
   decisive advantage once the evaluation was made causal and the playing field
   was levelled. That is the honest result.

Figures are written to `figures/` (`fig1_raw_data`, `fig2_decomposition`,
`fig3_predictions_h*`, `fig4_metrics_h*`).

---

## How it works

```
NAB CSV ──► physical [0,100]% scaling ──► per forecast origin:
                                          ├─ RevIN-normalize the 4-h window
                                          ├─ causal CEEMDAN of that window  ─► proposed model
                                          └─ (raw normalized window)        ─► baselines
            targets are the value `horizon` steps ahead, in real %.
```

- **Scaling** is the physical constant [0,100]% (CPU is bounded by definition),
  so it leaks nothing and never pushes a test value outside [0,1].
- **Decomposition** is cached per `(file, look_back, n_components, trials, seed)`
  and reused across horizons (it does not depend on the horizon).
- **Proposed model**: one CNN-BiLSTM sub-model per CEEMDAN component; the summed
  per-component forecast is trained end-to-end against the target (so nothing
  ever needs to decompose the future).
- **Baselines** (generic architecture families, not reproductions of any
  specific paper): LSTM, CNN-LSTM, Bi-LSTM, Transformer, + naive Persistence/EMA.

---

## Reproduce

```bash
python -m venv .venv && . .venv/Scripts/activate     # (Windows)  or  source .venv/bin/activate
pip install -r requirements.txt          # torch is pinned to 2.4.1 (see note in file)
python src/main.py                       # train + benchmark + export (CPU is fine)
pytest -q                                # 30 tests
```

Outputs: `results/metrics.csv`, `results/summary.txt`, `figures/*.png`, and
`model_export/` (served by the API). Knobs are env-overridable for a fast smoke
test, e.g. `CLOUDSENSE_HORIZONS=1 CLOUDSENSE_EPOCHS=3 CLOUDSENSE_TRIALS=3`.

The first run decomposes ~4000 causal windows (a few minutes) and caches them;
later runs and the second horizon are instant.

---

## Serve it

```bash
# 1) inference API (loads model_export/, reuses src/ feature code)
cd deploy/inference && PYTHONPATH=../../src uvicorn app:app --port 8000
#    POST /predict {"cpu_percent": [<=48 recent values>]} -> 4-hour forecast + scaling rec
#    GET  /health   GET /metrics

# 2) demo dashboard (proxies the API; "simulated fleet" is a local demo, not real AWS)
cd deploy/dashboard && CLOUDSENSE_INFERENCE_URL=http://127.0.0.1:8000 \
    uvicorn local_dashboard:app --port 8501
```

Docker: `docker build -f deploy/inference/Dockerfile -t cloudsense-api .` (build
from the repo root so `src/` and `model_export/` are included).

---

## What is real vs. designed

To keep the résumé/claims honest:

- **Built and runnable here:** the full training/benchmark pipeline; the FastAPI
  inference service (with the trained model); the local dashboard; a Dockerfile;
  optional AWS scripts (`deploy/scripts/`) and a CloudWatch→forecast→ASG
  integration (`deploy/scripts/cloudwatch_monitor.py`, needs your AWS creds).
- **Designed / illustrative:** the autoscaling "fleet" in the dashboard is a
  local simulation, and a full multi-service AWS architecture (API Gateway, ASG,
  etc.) is a *deployment design*, not something running in this repo.

---

## Limitations & future work

- Single NAB instance and a single seed (42); results will differ on other
  workloads. The reported numbers are verified **reproducible** (a re-run matches
  exactly), and a **multi-seed runner is provided** (`tools/robustness.py`) to
  produce mean ± std across seeds — a multi-instance study is future work.
- Causal CEEMDAN on a 48-point window yields relatively few stable IMFs; a longer
  causal decomposition context (e.g. 1 day) could help the proposed model and is
  worth testing.
- Persistence is the baseline to beat; promising directions are probabilistic
  forecasts (intervals for safer scaling) and explicitly modelling the daily
  seasonality the naive baselines exploit.

## Contributors

Built jointly by Apoorvan A and Rahul.

## License

MIT — see [LICENSE](LICENSE).
