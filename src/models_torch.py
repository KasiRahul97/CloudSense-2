"""
models_torch.py
===============
PyTorch models used in the CloudSense comparative study.

All learned models accept input of shape (batch, look_back, 1) and output
(batch, 1) -- a single value `horizon` steps ahead (direct forecasting).

Learned baselines (standard architecture families widely used for time-series
and cloud-workload forecasting). These are idiomatic implementations of each
family, NOT reproductions of any one paper's exact configuration, so they are
labelled generically and used as representative baselines on a common protocol:
1. LSTMModel        - stacked LSTM
2. CNNLSTMModel     - 1-D Conv + LSTM
3. BiLSTMModel      - bidirectional LSTM
4. TransformerModel - Transformer encoder
5. CEEMDANBiLSTM    - PROPOSED: one CNN-BiLSTM sub-model per signal component

Naive baselines (Persistence, EMA) live in train_evaluate.py since they need
no parameters or training.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Stacked LSTM baseline."""
    name = "LSTM"

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, look_back=48):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class CNNLSTMModel(nn.Module):
    """1-D conv feature extractor + stacked LSTM baseline."""
    name = "CNN-LSTM"

    def __init__(self, input_size=1, cnn_filters=32, kernel_size=3,
                 hidden_size=64, num_layers=2, dropout=0.2, look_back=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM baseline."""
    name = "Bi-LSTM"

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, look_back=48):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    """Transformer-encoder baseline for forecasting."""
    name = "Transformer"

    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, look_back=48):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=look_back + 10, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


# --------------------------------------------------------------------------
# PROPOSED: CEEMDAN + CNN-BiLSTM ensemble
# --------------------------------------------------------------------------
class IMFSubModel(nn.Module):
    """
    CNN-BiLSTM sub-model for a single signal component (IMF / residue).

    Conv1d(1->conv_filters) -> BiLSTM -> Linear. This exact architecture is
    mirrored in deploy/inference/app.py so exported weights load cleanly.
    """
    def __init__(self, look_back=48, hidden=32, conv_filters=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_filters, kernel_size=3, padding=1), nn.ReLU())
        self.bilstm = nn.LSTM(conv_filters, hidden, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        c = x.permute(0, 2, 1)
        c = self.conv(c).permute(0, 2, 1)
        out, _ = self.bilstm(c)
        return self.fc(out[:, -1, :])


class CEEMDANBiLSTM(nn.Module):
    """
    Proposed model: CEEMDAN decomposes the signal into `n_components` additive
    pieces; one IMFSubModel is trained per component; the final forecast is the
    sum of the per-component forecasts.

    The decomposition is performed in data_loader.decompose(); each sub-model
    is trained independently in train_evaluate.train_ceemdan_model().
    """
    name = "CEEMDAN+CNN-BiLSTM (Proposed)"

    def __init__(self, n_components=8, look_back=48, hidden=32, conv_filters=16):
        super().__init__()
        self.n_components = n_components
        self.look_back = look_back
        self.sub_models = nn.ModuleList(
            [IMFSubModel(look_back, hidden, conv_filters) for _ in range(n_components)])

    def forward(self, x_list):
        """x_list: list of n_components tensors (B, T, 1) -> (B, 1) summed."""
        preds = [sub(xi) for sub, xi in zip(self.sub_models, x_list)]
        return torch.stack(preds, dim=0).sum(dim=0)
