"""
train_evaluate.py
=================
Training and evaluation for the CloudSense benchmark.

Correctness / fairness guarantees
---------------------------------
* CAUSAL, LEAKAGE-FREE features. Every input window is RevIN-normalized and
  (for the proposed model) CEEMDAN-decomposed using ONLY that past window
  (data_loader.window_components). Nothing peeks at the future, and the exact
  same transform runs in deploy/inference -> benchmark == production.
* LEVEL-INVARIANT targets. Models predict the RevIN-normalized target
  yn = (y - mu) / sd; predictions are denormalized back to raw CPU-% before any
  metric is computed, so MAE/RMSE/MAPE/R2 are all in real %.
* ONE shared ground truth. Naive baselines, learned baselines and the proposed
  ensemble are scored against the identical per-origin test targets.
* Naive baselines (Persistence, EMA) are the essential sanity check: a learned
  model that cannot beat persistence is not useful.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mape(y_true, y_pred, eps=1.0):
    # eps in CPU-% units avoids the divide-by-near-zero blow-up of raw MAPE.
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_metrics(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


# ---------------------------------------------------------------------------
# Scaling helpers (raw % <-> scaled [0,1] <-> RevIN-normalized)
# ---------------------------------------------------------------------------
def _to_pct(scaled_vals, scaler):
    return scaler.inverse_transform(np.asarray(scaled_vals).reshape(-1, 1)).flatten()


def test_targets(data):
    """The single ground-truth array (raw %) all models are scored against."""
    return _to_pct(data["splits"]["test"]["y_scaled"], data["scaler"])


def _revin_apply(X_scaled, mu, sd):
    """(N, L) scaled windows -> RevIN-normalized windows using per-row (mu, sd)."""
    return (X_scaled - mu[:, None]) / sd[:, None]


def _denorm_to_pct(pred_norm, mu, sd, scaler):
    """RevIN-normalized prediction -> raw CPU-% (undo RevIN, clip, undo scaling)."""
    pred_scaled = np.clip(pred_norm * sd + mu, 0.0, 1.0)
    return _to_pct(pred_scaled, scaler)


# ---------------------------------------------------------------------------
# Naive baselines (no training) -- the critical sanity check
# ---------------------------------------------------------------------------
def evaluate_persistence(data):
    """Predict the value `horizon` steps ahead as the last observed value."""
    te = data["splits"]["test"]
    preds = _to_pct(np.clip(te["X_scaled"][:, -1], 0, 1), data["scaler"])
    y_true = test_targets(data)
    return preds, y_true, compute_metrics(y_true, preds)


def evaluate_ema(data, alpha=0.3):
    """Exponential moving average of the input window as the forecast."""
    te = data["splits"]["test"]
    X = te["X_scaled"]
    w = (1 - alpha) ** np.arange(X.shape[1])[::-1]
    w = w / w.sum()
    preds = _to_pct(np.clip(X @ w, 0, 1), data["scaler"])
    y_true = test_targets(data)
    return preds, y_true, compute_metrics(y_true, preds)


# ---------------------------------------------------------------------------
# Shared training loop (works for both the single-tensor baselines and the
# list-input ensemble, via a `predict(model, Xb)` callable)
# ---------------------------------------------------------------------------
def _fit(model, tr_dl, va_dl, predict, epochs, lr, patience, verbose):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=7)
    crit = nn.MSELoss()
    best_val, best_state, no_improv = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        tl = []
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(predict(model, Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for Xb, yb in va_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                vl.append(crit(predict(model, Xb), yb).item())

        t_loss = float(np.mean(tl))
        v_loss = float(np.mean(vl)) if vl else float("inf")
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        sch.step(v_loss)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improv = 0
        else:
            no_improv += 1
        if verbose and ep % 20 == 0:
            print(f"      epoch {ep:3d} | train={t_loss:.5f} val={v_loss:.5f}")
        if no_improv >= patience:
            if verbose:
                print(f"      early stop @ epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _loaders(X, y, batch_size, seed):
    Xt = torch.FloatTensor(X)
    yt = torch.FloatTensor(y).unsqueeze(-1)
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True, generator=g)
    return tr


# ---------------------------------------------------------------------------
# Standard learned models (RevIN-normalized single-channel windows)
# ---------------------------------------------------------------------------
def _predict_standard(model, Xb):
    return model(Xb)                                  # Xb: (B, L, 1)


def train_model(model, data, epochs=120, lr=1e-3, batch_size=64,
                patience=15, seed=42, verbose=True):
    set_seed(seed)
    model = model.to(DEVICE)
    tr, va = data["splits"]["train"], data["splits"]["val"]

    Xtr = _revin_apply(tr["X_scaled"], tr["mu"], tr["sd"])[..., None]   # (N, L, 1)
    ytr = (tr["y_scaled"] - tr["mu"]) / tr["sd"]
    Xva = _revin_apply(va["X_scaled"], va["mu"], va["sd"])[..., None]
    yva = (va["y_scaled"] - va["mu"]) / va["sd"]

    tr_dl = _loaders(Xtr, ytr, batch_size, seed)
    va_dl = _loaders(Xva, yva, batch_size * 2, seed)
    return _fit(model, tr_dl, va_dl, _predict_standard, epochs, lr, patience, verbose)


def evaluate_model(model, data):
    model.eval()
    te = data["splits"]["test"]
    Xte = _revin_apply(te["X_scaled"], te["mu"], te["sd"])[..., None]
    Xt = torch.FloatTensor(Xte).to(DEVICE)
    with torch.no_grad():
        pred_norm = model(Xt).cpu().numpy().flatten()
    preds = _denorm_to_pct(pred_norm, te["mu"], te["sd"], data["scaler"])
    y_true = test_targets(data)
    return preds, y_true, compute_metrics(y_true, preds)


# ---------------------------------------------------------------------------
# Proposed CEEMDAN + CNN-BiLSTM ensemble (END-TO-END, causal components)
# ---------------------------------------------------------------------------
def _components_to_list(Xc):
    """(B, C, L) -> list of C tensors (B, L, 1) for CEEMDANBiLSTM.forward."""
    return [Xc[:, k, :].unsqueeze(-1) for k in range(Xc.shape[1])]


def _predict_ensemble(model, Xb):
    return model(_components_to_list(Xb))             # Xb: (B, C, L)


def train_ceemdan_model(model, data, epochs=120, lr=1e-3, batch_size=64,
                        patience=15, seed=42, verbose=True):
    """Train the WHOLE ensemble end-to-end: the summed per-component forecast is
    fit directly to the (normalized) target. No per-component target is ever
    constructed, so nothing requires decomposing the future -> fully causal and
    identical to what deploy/inference computes."""
    set_seed(seed)
    model = model.to(DEVICE)
    tr, va = data["splits"]["train"], data["splits"]["val"]

    ytr = (tr["y_scaled"] - tr["mu"]) / tr["sd"]
    yva = (va["y_scaled"] - va["mu"]) / va["sd"]
    tr_dl = _loaders(tr["Xc"], ytr, batch_size, seed)   # Xc: (N, C, L)
    va_dl = _loaders(va["Xc"], yva, batch_size * 2, seed)
    return _fit(model, tr_dl, va_dl, _predict_ensemble, epochs, lr, patience, verbose)


def evaluate_ceemdan_model(model, data):
    model.eval()
    te = data["splits"]["test"]
    Xc = torch.FloatTensor(te["Xc"]).to(DEVICE)
    with torch.no_grad():
        pred_norm = model(_components_to_list(Xc)).cpu().numpy().flatten()
    preds = _denorm_to_pct(pred_norm, te["mu"], te["sd"], data["scaler"])
    y_true = test_targets(data)
    return preds, y_true, compute_metrics(y_true, preds)
