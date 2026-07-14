"""Tests for model forward shapes and the proposed ensemble's structure."""

import torch
import pytest

import models_torch as mt

B, T = 4, 48


@pytest.mark.parametrize("factory", [
    lambda: mt.LSTMModel(look_back=T),
    lambda: mt.CNNLSTMModel(look_back=T),
    lambda: mt.BiLSTMModel(look_back=T),
    lambda: mt.TransformerModel(look_back=T),
])
def test_standard_model_forward_shape(factory):
    model = factory().eval()
    x = torch.randn(B, T, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, 1)
    assert torch.isfinite(out).all()


def test_imf_submodel_forward_shape():
    sub = mt.IMFSubModel(look_back=T, hidden=32, conv_filters=16).eval()
    x = torch.randn(B, T, 1)
    with torch.no_grad():
        out = sub(x)
    assert out.shape == (B, 1)


def test_proposed_ensemble_structure_and_sum():
    n = 8
    model = mt.CEEMDANBiLSTM(n_components=n, look_back=T, hidden=32, conv_filters=16).eval()
    assert len(model.sub_models) == n
    xs = [torch.randn(B, T, 1) for _ in range(n)]
    with torch.no_grad():
        out = model(xs)
    assert out.shape == (B, 1)
    # forward() is the sum of the per-component sub-model outputs.
    with torch.no_grad():
        manual = sum(sub(xi) for sub, xi in zip(model.sub_models, xs))
    torch.testing.assert_close(out, manual)
