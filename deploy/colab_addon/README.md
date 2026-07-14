# Training on Colab (optional)

Training is **self-contained** in `src/main.py` and runs fine on CPU
(torch 2.4.1). One command trains every model, writes the benchmark, and
exports the served model:

```bash
python src/main.py            # -> results/, figures/, model_export/
```

`model_export/` (weights + `model_config.json` + `scaler.pkl` +
`training_metrics.json`) is exactly what `deploy/inference` loads, so there is
**no separate manual export step** anymore (the old `export_model.py` Colab cell
is removed).

To use a free GPU for speed, run the same `src/main.py` in a Colab notebook with
a GPU runtime — the code auto-detects CUDA (`train_evaluate.DEVICE`). Then
download the `model_export/` folder and point the API at it via `MODEL_DIR`.
