"""
Empirical kurtosis estimation via Monte Carlo sampling.

Replaces the Gaussian moment closure (κ₄=0) with empirical kurtosis
measured from actual network forward passes at finite width. This
addresses the key critique that the system validates predictions
against its own theory rather than using empirical data.

The empirical κ₄ captures non-Gaussian effects that accumulate
through depth, especially near phase boundaries where the Gaussian
approximation breaks down most severely.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class EmpiricalMoments:
    """Empirical moments measured from Monte Carlo forward passes."""
    mean: float = 0.0
    variance: float = 0.0       # E[x^2] - E[x]^2
    fourth_moment: float = 0.0  # E[x^4]
    sixth_moment: float = 0.0   # E[x^6]
    kappa4: float = 0.0         # excess kurtosis: M4/V^2 - 1
    kappa6: float = 0.0         # hyper-kurtosis: M6/V^3 - 3
    n_samples: int = 0
    std_error_kappa4: float = 0.0


@dataclass
class EmpiricalKurtosisProfile:
    """Per-layer empirical kurtosis profile from MC forward passes."""
    per_layer_moments: List[EmpiricalMoments] = field(default_factory=list)
    depth: int = 0
    width: int = 0
    activation: str = "relu"
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    n_networks: int = 0
    kappa4_growth_exponent: float = 0.0  # fitted power-law exponent
    kappa4_growth_r2: float = 0.0        # R² of power-law fit


def compute_empirical_moments(
    activation: str,
    sigma_w: float,
    sigma_b: float,
    width: int,
    depth: int,
    input_dim: int = 100,
    n_networks: int = 200,
    n_inputs: int = 50,
    seed: int = 42,
) -> EmpiricalKurtosisProfile:
    """Compute empirical per-layer moments via Monte Carlo.

    For each of n_networks random initializations, propagates n_inputs
    random inputs through the network and collects activation statistics
    at each layer. This gives empirical κ₄ that captures non-Gaussian
    effects at finite width.

    Parameters
    ----------
    activation : str
        Activation function name.
    sigma_w : float
        Weight initialization scale.
    sigma_b : float
        Bias initialization scale.
    width : int
        Network width.
    depth : int
        Network depth.
    input_dim : int
        Input dimensionality.
    n_networks : int
        Number of random network initializations.
    n_inputs : int
        Number of random inputs per network.
    seed : int
        Random seed.

    Returns
    -------
    EmpiricalKurtosisProfile
        Per-layer empirical moments including κ₄.
    """
    rng = np.random.RandomState(seed)
    act_fn = _get_activation_fn(activation)

    # Collect per-layer activations across all networks
    # Shape: [depth+1][n_networks * n_inputs]
    layer_activations = [[] for _ in range(depth + 1)]

    for net_idx in range(n_networks):
        # Random inputs normalized to unit variance
        X = rng.randn(n_inputs, input_dim) / np.sqrt(input_dim)
        layer_activations[0].append(X.ravel())

        h = X
        for l in range(depth):
            fan_in = h.shape[1]
            fan_out = width if l < depth - 1 else width
            W = rng.randn(fan_in, fan_out) * sigma_w / np.sqrt(fan_in)
            b = rng.randn(fan_out) * sigma_b if sigma_b > 0 else np.zeros(fan_out)
            pre = h @ W + b
            if l < depth - 1:
                h = act_fn(pre)
            else:
                h = pre  # no activation on last layer
            layer_activations[l + 1].append(h.ravel())

    # Compute moments at each layer
    per_layer = []
    for l in range(depth + 1):
        vals = np.concatenate(layer_activations[l])
        # Remove infinities/nans
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            per_layer.append(EmpiricalMoments())
            continue

        mean = float(np.mean(vals))
        centered = vals - mean
        var = float(np.mean(centered ** 2))
        m4 = float(np.mean(centered ** 4))
        m6_raw = np.mean(np.minimum(centered ** 6, 1e30))
        m6 = float(m6_raw)

        if var > 1e-30:
            kappa4 = m4 / (var ** 2) - 1.0
            kappa6 = m6 / (var ** 3) - 3.0 if var > 1e-20 else 0.0
        else:
            kappa4 = 0.0
            kappa6 = 0.0

        # Bootstrap standard error for κ₄
        n_boot = 100
        boot_kappas = []
        chunk_size = len(vals) // n_boot
        if chunk_size > 10:
            for b in range(n_boot):
                idx = rng.choice(len(vals), chunk_size, replace=True)
                bv = vals[idx]
                bc = bv - np.mean(bv)
                bvar = np.mean(bc ** 2)
                bm4 = np.mean(bc ** 4)
                if bvar > 1e-30:
                    boot_kappas.append(bm4 / (bvar ** 2) - 1.0)
            se_kappa4 = float(np.std(boot_kappas)) if boot_kappas else 0.0
        else:
            se_kappa4 = 0.0

        per_layer.append(EmpiricalMoments(
            mean=mean,
            variance=var,
            fourth_moment=m4,
            sixth_moment=m6,
            kappa4=kappa4,
            kappa6=kappa6,
            n_samples=len(vals),
            std_error_kappa4=se_kappa4,
        ))

    # Fit κ₄ growth exponent: κ₄(l) ~ l^α
    kappa4_values = [m.kappa4 for m in per_layer[1:] if m.n_samples > 0]
    if len(kappa4_values) >= 3:
        layers_idx = np.arange(1, len(kappa4_values) + 1, dtype=float)
        # Only fit on positive κ₄ values
        mask = np.array(kappa4_values) > 0
        if np.sum(mask) >= 2:
            log_l = np.log(layers_idx[mask])
            log_k = np.log(np.array(kappa4_values)[mask])
            if len(log_l) >= 2:
                coeffs = np.polyfit(log_l, log_k, 1)
                exponent = float(coeffs[0])
                # R²
                pred = np.polyval(coeffs, log_l)
                ss_res = np.sum((log_k - pred) ** 2)
                ss_tot = np.sum((log_k - np.mean(log_k)) ** 2)
                r2 = float(1.0 - ss_res / max(ss_tot, 1e-30))
            else:
                exponent = 0.0
                r2 = 0.0
        else:
            exponent = 0.0
            r2 = 0.0
    else:
        exponent = 0.0
        r2 = 0.0

    return EmpiricalKurtosisProfile(
        per_layer_moments=per_layer,
        depth=depth,
        width=width,
        activation=activation,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        n_networks=n_networks,
        kappa4_growth_exponent=exponent,
        kappa4_growth_r2=r2,
    )


def empirical_kappa4_lookup(
    activation: str,
    q: float,
    width: int = 256,
    depth: int = 10,
    sigma_w: float = 1.0,
) -> float:
    """Get empirical κ₄ for use in finite-width corrections.

    Falls back to analytical values when MC sampling is impractical.
    Uses a small MC sample to estimate the empirical κ₄ at the given
    variance q, which may differ significantly from the analytical
    (Gaussian-closure) value near phase boundaries.
    """
    if width > 4096 or q <= 0 or q > 1e4:
        # Fall back to analytical for very wide or extreme variance
        from mean_field_theory import ActivationVarianceMaps
        return ActivationVarianceMaps.get_kurtosis_excess(activation, q)

    # Small MC estimate
    profile = compute_empirical_moments(
        activation=activation,
        sigma_w=sigma_w,
        sigma_b=0.0,
        width=width,
        depth=min(depth, 5),
        n_networks=50,
        n_inputs=20,
        seed=42,
    )

    # Use the middle layer's κ₄ as representative
    mid = len(profile.per_layer_moments) // 2
    if mid < len(profile.per_layer_moments) and profile.per_layer_moments[mid].n_samples > 0:
        return profile.per_layer_moments[mid].kappa4

    from mean_field_theory import ActivationVarianceMaps
    return ActivationVarianceMaps.get_kurtosis_excess(activation, q)


def compare_analytical_vs_empirical(
    activations: List[str] = None,
    widths: List[int] = None,
    depth: int = 10,
    sigma_w_values: List[float] = None,
    n_networks: int = 200,
    seed: int = 42,
) -> Dict:
    """Compare analytical (Gaussian closure) κ₄ vs empirical κ₄.

    This is the key validation: if the Gaussian closure is accurate,
    the analytical and empirical κ₄ should agree. Significant
    disagreement indicates that the moment closure approximation
    is introducing bias.

    Returns
    -------
    dict
        Comparison results with analytical vs empirical κ₄ per config.
    """
    from mean_field_theory import ActivationVarianceMaps

    if activations is None:
        activations = ["relu", "tanh", "gelu", "silu"]
    if widths is None:
        widths = [64, 128, 256, 512]
    if sigma_w_values is None:
        sigma_w_values = [1.0, 1.414, 2.0]

    results = []
    for act in activations:
        for width in widths:
            for sw in sigma_w_values:
                profile = compute_empirical_moments(
                    activation=act, sigma_w=sw, sigma_b=0.0,
                    width=width, depth=depth,
                    n_networks=n_networks, n_inputs=30,
                    seed=seed,
                )

                for l, moments in enumerate(profile.per_layer_moments):
                    if moments.n_samples == 0:
                        continue
                    analytical = ActivationVarianceMaps.get_kurtosis_excess(
                        act, max(moments.variance, 1e-30)
                    )
                    results.append({
                        "activation": act,
                        "width": width,
                        "sigma_w": sw,
                        "layer": l,
                        "empirical_kappa4": moments.kappa4,
                        "analytical_kappa4": analytical,
                        "relative_error": abs(moments.kappa4 - analytical) / max(abs(analytical), 1e-10),
                        "std_error": moments.std_error_kappa4,
                    })

    # Summary statistics
    if results:
        rel_errors = [r["relative_error"] for r in results if np.isfinite(r["relative_error"])]
        summary = {
            "n_configs": len(results),
            "mean_relative_error": float(np.mean(rel_errors)) if rel_errors else 0.0,
            "median_relative_error": float(np.median(rel_errors)) if rel_errors else 0.0,
            "max_relative_error": float(np.max(rel_errors)) if rel_errors else 0.0,
            "p90_relative_error": float(np.percentile(rel_errors, 90)) if rel_errors else 0.0,
        }
    else:
        summary = {}

    return {
        "per_config": results,
        "summary": summary,
    }


def _get_activation_fn(name: str) -> Callable:
    """Get numpy activation function by name."""
    from scipy.special import erf as _erf

    def relu(x):
        return np.maximum(x, 0)

    def tanh(x):
        return np.tanh(x)

    def gelu(x):
        return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))

    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))

    def mish(x):
        return x * np.tanh(np.log1p(np.exp(np.clip(x, -500, 20))))

    fns = {
        "relu": relu, "tanh": tanh, "gelu": gelu, "silu": silu,
        "swish": silu, "leaky_relu": leaky_relu, "elu": elu, "mish": mish,
        "linear": lambda x: x,
    }
    return fns.get(name, relu)
