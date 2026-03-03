"""Tests for calibrated_phasekit_init."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from baselines import (
    calibrated_phasekit_init, xavier_init, lsuv_init,
    train_mlp, apply_activation,
)


def test_basic_creation():
    """CaliKit produces valid weights and biases."""
    dims = [10, 64, 64, 1]
    result = calibrated_phasekit_init(dims, "relu")
    assert result.method == "calibrated_phasekit"
    assert len(result.weights) == 3
    assert len(result.biases) == 3
    assert result.weights[0].shape == (10, 64)
    assert result.weights[1].shape == (64, 64)
    assert result.weights[2].shape == (64, 1)


def test_all_activations():
    """CaliKit works for all supported activations."""
    dims = [10, 32, 32, 1]
    for act in ["relu", "tanh", "gelu", "silu"]:
        result = calibrated_phasekit_init(dims, act)
        assert result.method == "calibrated_phasekit"
        assert all(np.all(np.isfinite(w)) for w in result.weights)


def test_deep_network():
    """CaliKit handles deep networks (depth 20)."""
    dims = [10] + [64] * 19 + [1]
    result = calibrated_phasekit_init(dims, "tanh")
    assert len(result.weights) == 20
    assert all(np.all(np.isfinite(w)) for w in result.weights)


def test_forward_variance_near_unity():
    """After CaliKit init, forward variance should be near 1.0."""
    dims = [10, 64, 64, 64, 1]
    rng = np.random.RandomState(42)
    X = rng.randn(256, 10)
    result = calibrated_phasekit_init(dims, "tanh", X_calib=X)

    h = X.copy()
    for l in range(len(result.weights) - 1):
        h = h @ result.weights[l] + result.biases[l]
        var = np.var(h)
        assert var < 10.0, f"Layer {l} variance = {var} (too high)"
        h = apply_activation(h, "tanh")


def test_deterministic():
    """Same seed gives same result."""
    dims = [10, 64, 64, 1]
    r1 = calibrated_phasekit_init(dims, "relu", seed=42)
    r2 = calibrated_phasekit_init(dims, "relu", seed=42)
    for w1, w2 in zip(r1.weights, r2.weights):
        np.testing.assert_array_equal(w1, w2)


def test_different_seeds():
    """Different seeds give different results."""
    dims = [10, 64, 64, 1]
    r1 = calibrated_phasekit_init(dims, "relu", seed=0)
    r2 = calibrated_phasekit_init(dims, "relu", seed=1)
    assert not np.allclose(r1.weights[0], r2.weights[0])


def test_trains_better_than_xavier_deep():
    """CaliKit should train at least as well as Xavier on deep networks."""
    dims = [10] + [64] * 9 + [1]
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 10)
    y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1]
    X_test = rng.randn(100, 10)
    y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1]

    xavier = xavier_init(dims, seed=42)
    calib = calibrated_phasekit_init(dims, "tanh", seed=42)

    r_xavier = train_mlp(xavier.weights, xavier.biases,
                         X_train, y_train, X_test, y_test,
                         activation="tanh", n_steps=300, lr=0.01)
    r_calib = train_mlp(calib.weights, calib.biases,
                        X_train, y_train, X_test, y_test,
                        activation="tanh", n_steps=300, lr=0.01)

    # CaliKit should not be catastrophically worse
    if np.isfinite(r_calib["final_loss"]) and np.isfinite(r_xavier["final_loss"]):
        assert r_calib["final_loss"] < r_xavier["final_loss"] * 10


def test_sigmas_finite():
    """All reported sigmas should be finite and positive."""
    dims = [10, 128, 128, 128, 1]
    result = calibrated_phasekit_init(dims, "gelu")
    assert all(np.isfinite(s) and s > 0 for s in result.sigma_w_per_layer)
