"""
PhaseKit: Zero-cost initialization diagnostics for any PyTorch model.

The primary user-facing API. Three functions cover all use cases:

    from phasekit import classify, initialize, diagnose

    # 1. Classify phase (ordered / critical / chaotic)
    result = classify(model, input_shape=(3, 32, 32))

    # 2. Initialize to edge-of-chaos
    initialize(model, input_shape=(3, 32, 32))

    # 3. Full diagnostic report
    report = diagnose(model, input_shape=(3, 32, 32))

Works on MLPs, CNNs, ResNets, Transformers, and arbitrary nn.Modules.
Uses DAG-based variance propagation with torch.fx symbolic tracing.
No training data or gradient computation required.

References:
    Poole et al., "Exponential expressivity", NeurIPS 2016
    Schoenholz et al., "Deep Information Propagation", ICLR 2017
    Noci et al., "Signal Propagation in Transformers", ICML 2022
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import internal modules
try:
    from dag_propagator import (
        analyze_dag,
        DAGAnalysisResult,
        DAGBuilder,
        DAGVariancePropagator,
        recommend_sigma_w_dag,
        trace_empirical_variance,
        compute_dag_chi,
        _estimate_width,
    )
    from mean_field_theory import (
        MeanFieldAnalyzer,
        ArchitectureSpec,
        MFReport,
        ActivationVarianceMaps,
    )
    from transformer_mean_field import (
        TransformerMeanField,
        TransformerSpec,
        TransformerMFReport,
    )
except ImportError:
    from .dag_propagator import (
        analyze_dag,
        DAGAnalysisResult,
        DAGBuilder,
        DAGVariancePropagator,
        recommend_sigma_w_dag,
        trace_empirical_variance,
        compute_dag_chi,
        _estimate_width,
    )
    from .mean_field_theory import (
        MeanFieldAnalyzer,
        ArchitectureSpec,
        MFReport,
        ActivationVarianceMaps,
    )
    from .transformer_mean_field import (
        TransformerMeanField,
        TransformerSpec,
        TransformerMFReport,
    )


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ClassifyResult:
    """Result of phase classification.

    Attributes
    ----------
    phase : str
        'ordered', 'critical', or 'chaotic'.
    chi_total : float
        Effective total susceptibility.
    depth_scale : float
        1 / |ln(chi_total)|. Inf at criticality.
    variance_error_pct : float
        Mean relative error between predicted and empirical variance (%).
    recommended_sigma_w : float
        σ_w for edge-of-chaos initialization.
    arch_summary : str
        Architecture description string.
    n_weight_layers : int
        Number of weight (Linear/Conv) layers.
    n_params : int
        Total trainable parameters.
    has_attention : bool
    has_normalization : bool
    has_residual : bool
    per_layer_chi : dict
        Per-node chi values.
    explanation : str
        Human-readable analysis.
    """
    phase: str = "unknown"
    chi_total: float = 1.0
    depth_scale: float = float('inf')
    variance_error_pct: float = 0.0
    recommended_sigma_w: float = 1.0
    arch_summary: str = ""
    n_weight_layers: int = 0
    n_params: int = 0
    has_attention: bool = False
    has_normalization: bool = False
    has_residual: bool = False
    per_layer_chi: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class InitResult:
    """Result of initialization.

    Attributes
    ----------
    phase_before : str
        Phase before initialization.
    phase_after : str
        Phase after initialization.
    chi_before : float
    chi_after : float
    layers_modified : int
        Number of layers whose weights were rescaled.
    sigma_w_applied : dict
        Per-layer σ_w values that were applied.
    """
    phase_before: str = "unknown"
    phase_after: str = "unknown"
    chi_before: float = 1.0
    chi_after: float = 1.0
    layers_modified: int = 0
    sigma_w_applied: Dict[str, float] = field(default_factory=dict)


@dataclass
class DiagnoseResult:
    """Full diagnostic report.

    Attributes
    ----------
    classification : ClassifyResult
    dag_result : DAGAnalysisResult
        Full DAG analysis with per-node variances.
    recommendations : list of str
        Actionable recommendations.
    warnings : list of str
        Potential issues detected.
    """
    classification: ClassifyResult = field(default_factory=ClassifyResult)
    dag_result: Optional[DAGAnalysisResult] = None
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ======================================================================
# Core API
# ======================================================================

def classify(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    n_samples: int = 256,
    seed: int = 42,
    seq_len: int = 128,
    finite_width: bool = True,
) -> ClassifyResult:
    """Classify the initialization phase of any PyTorch model.

    Builds a DAG of the computation graph, propagates second-moment
    statistics through topologically sorted nodes, and classifies
    the phase as ordered, critical, or chaotic.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    input_shape : tuple
        Shape of a single input (no batch dimension).
        E.g., (784,) for MLP, (3, 32, 32) for CNN, (128, 512) for Transformer.
    n_samples : int
        Batch size for empirical variance measurement.
    seed : int
        Random seed.
    seq_len : int
        Sequence length for attention layers.
    finite_width : bool
        Whether to apply O(1/N) finite-width corrections.

    Returns
    -------
    ClassifyResult

    Examples
    --------
    >>> model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    >>> result = classify(model, input_shape=(784,))
    >>> print(result.phase, result.chi_total)
    """
    _require_torch()

    dag = analyze_dag(
        model, input_shape,
        n_samples=n_samples, seed=seed,
        seq_len=seq_len,
        apply_finite_width=finite_width,
    )

    chi = dag.chi_total
    if abs(chi) > 1e-12 and chi > 0:
        depth_scale = 1.0 / abs(math.log(max(chi, 1e-30)))
    else:
        depth_scale = float('inf')

    return ClassifyResult(
        phase=dag.phase,
        chi_total=chi,
        depth_scale=depth_scale,
        variance_error_pct=dag.variance_error_pct,
        recommended_sigma_w=dag.recommended_sigma_w,
        arch_summary=dag.architecture_summary,
        n_weight_layers=dag.n_weight_layers,
        n_params=dag.n_params,
        has_attention=dag.has_attention,
        has_normalization=dag.has_norm,
        has_residual=dag.has_residual,
        per_layer_chi=dag.chi_per_node,
        explanation=dag.explanation,
    )


def initialize(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    target_chi: float = 1.0,
    strategy: str = "per_layer",
    n_samples: int = 256,
    seed: int = 42,
    seq_len: int = 128,
) -> InitResult:
    """Initialize a model to the edge of chaos (or a target chi).

    Uses DAG analysis to compute per-layer σ_w recommendations,
    then rescales weight matrices to achieve the target susceptibility.

    Parameters
    ----------
    model : nn.Module
        Model to initialize (modified in-place).
    input_shape : tuple
        Shape of a single input.
    target_chi : float
        Target per-block susceptibility. Default 1.0 (edge of chaos).
        For transformers, exp(1/L) is often better.
    strategy : str
        'per_layer': rescale each layer independently.
        'global': apply a single scale factor to all layers.
        'depth_aware': slightly ordered for deep nets to prevent explosion.
    n_samples : int
        Batch size for empirical variance measurement.
    seed : int
    seq_len : int

    Returns
    -------
    InitResult

    Examples
    --------
    >>> model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    >>> result = initialize(model, input_shape=(784,))
    >>> print(result.phase_after, result.chi_after)
    """
    _require_torch()

    # Phase before
    before = classify(model, input_shape, n_samples, seed, seq_len)

    # Get per-layer recommendations from DAG
    dag = analyze_dag(
        model, input_shape,
        n_samples=n_samples, seed=seed, seq_len=seq_len,
    )

    # Determine depth for depth-aware strategy
    depth = dag.n_weight_layers
    if strategy == "depth_aware" and depth > 5:
        # For deep networks, target slightly ordered to prevent explosion
        target_chi = 1.0 - 0.5 * math.log(depth) / depth

    # Detect if transformer
    is_transformer = dag.has_attention
    if is_transformer and target_chi == 1.0:
        n_blocks = max(1, sum(1 for n in dag.nodes.values() if n.is_attention))
        target_chi = math.exp(1.0 / max(n_blocks, 1))

    # Get recommended sigma_w per layer
    recs = dag.per_layer_recommendations
    sigma_w_applied = {}
    layers_modified = 0

    # Detect activation function used in the model
    detected_act = 'relu'
    for m in model.modules():
        cls = type(m).__name__
        act_map = {'ReLU': 'relu', 'GELU': 'gelu', 'SiLU': 'silu',
                   'Tanh': 'tanh', 'LeakyReLU': 'leaky_relu', 'ELU': 'elu', 'Mish': 'mish'}
        if cls in act_map:
            detected_act = act_map[cls]
            break

    if strategy == "global":
        # Single global scale factor
        if before.chi_total > 1e-12:
            scale = math.sqrt(target_chi / before.chi_total)
        else:
            scale = 1.0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                with torch.no_grad():
                    module.weight.mul_(scale)
                sigma_w_applied[name] = scale
                layers_modified += 1
    else:
        # Theory-driven initialization: set sigma_w for criticality at q₀=1
        sw_star = edge_of_chaos_sigma_w(detected_act, at_init=True)

        # For deep networks, apply depth correction
        if strategy == "depth_aware" and depth > 5:
            # Target slightly below critical to prevent cumulative explosion
            depth_correction = 1.0 - 0.01 * math.log(depth)
            sw_target = sw_star * max(depth_correction, 0.9)
        else:
            sw_target = sw_star

        # For transformers: don't modify attention/norm layers
        is_transformer = dag.has_attention

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Skip norm/attention-related layers in transformers
                if is_transformer and any(k in name for k in ['ln', 'norm', 'layer_norm']):
                    continue

                fan_in = module.weight.shape[1] if module.weight.dim() >= 2 else module.weight.shape[0]
                if module.weight.dim() > 2:
                    fan_in = int(np.prod(module.weight.shape[1:]))

                # Apply sw_target / sqrt(fan_in) initialization
                with torch.no_grad():
                    nn.init.normal_(module.weight, 0, sw_target / math.sqrt(fan_in))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                sigma_w_applied[name] = sw_target
                layers_modified += 1

    # Phase after
    after = classify(model, input_shape, n_samples, seed, seq_len)

    return InitResult(
        phase_before=before.phase,
        phase_after=after.phase,
        chi_before=before.chi_total,
        chi_after=after.chi_total,
        layers_modified=layers_modified,
        sigma_w_applied=sigma_w_applied,
    )


def diagnose(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    n_samples: int = 256,
    seed: int = 42,
    seq_len: int = 128,
) -> DiagnoseResult:
    """Full diagnostic report for a model's initialization.

    Performs DAG analysis, phase classification, and generates
    actionable recommendations and warnings.

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple
    n_samples : int
    seed : int
    seq_len : int

    Returns
    -------
    DiagnoseResult

    Examples
    --------
    >>> model = build_my_model()
    >>> report = diagnose(model, input_shape=(3, 32, 32))
    >>> for r in report.recommendations:
    ...     print(r)
    """
    _require_torch()

    dag = analyze_dag(
        model, input_shape,
        n_samples=n_samples, seed=seed, seq_len=seq_len,
    )

    chi = dag.chi_total
    if abs(chi) > 1e-12 and chi > 0:
        depth_scale = 1.0 / abs(math.log(max(chi, 1e-30)))
    else:
        depth_scale = float('inf')

    cls_result = ClassifyResult(
        phase=dag.phase,
        chi_total=chi,
        depth_scale=depth_scale,
        variance_error_pct=dag.variance_error_pct,
        recommended_sigma_w=dag.recommended_sigma_w,
        arch_summary=dag.architecture_summary,
        n_weight_layers=dag.n_weight_layers,
        n_params=dag.n_params,
        has_attention=dag.has_attention,
        has_normalization=dag.has_norm,
        has_residual=dag.has_residual,
        per_layer_chi=dag.chi_per_node,
        explanation=dag.explanation,
    )

    # Generate recommendations
    recs = []
    warns = []

    if dag.phase == "chaotic":
        recs.append(
            f"Network is chaotic (χ={chi:.3f}). Reduce σ_w to ~{dag.recommended_sigma_w:.4f} "
            f"to reach edge of chaos. Gradients will explode after ~{depth_scale:.0f} layers."
        )
    elif dag.phase == "ordered":
        recs.append(
            f"Network is ordered (χ={chi:.3f}). Increase σ_w to ~{dag.recommended_sigma_w:.4f}. "
            f"Gradients vanish after ~{depth_scale:.0f} layers."
        )
    else:
        recs.append(
            f"Network is near-critical (χ={chi:.3f}). Current initialization is good. "
            f"Depth scale ≈ {depth_scale:.0f} layers."
        )

    if dag.n_weight_layers > 20 and not dag.has_norm and not dag.has_residual:
        warns.append(
            f"Deep network ({dag.n_weight_layers} layers) without normalization or skip connections. "
            f"Consider adding LayerNorm/BatchNorm or residual connections for stable training."
        )

    if dag.variance_error_pct > 50:
        warns.append(
            f"High variance prediction error ({dag.variance_error_pct:.1f}%). "
            f"Mean-field approximation may be inaccurate for this architecture."
        )

    # Per-layer variance anomalies
    emp_vals = list(dag.empirical_variance.values())
    if emp_vals:
        max_var = max(emp_vals)
        min_var = min(v for v in emp_vals if v > 0)
        if max_var > 1e6 * min_var:
            warns.append(
                f"Variance spans {max_var/min_var:.0e}× across layers. "
                f"Some layers may have exploding/vanishing activations."
            )

    # Transformer-specific
    if dag.has_attention:
        n_attn = sum(1 for n in dag.nodes.values() if n.is_attention)
        recs.append(
            f"Transformer with {n_attn} attention layers detected. "
            f"For best results, target χ_block = exp(1/{n_attn}) ≈ {math.exp(1.0/max(n_attn,1)):.4f}."
        )

    return DiagnoseResult(
        classification=cls_result,
        dag_result=dag,
        recommendations=recs,
        warnings=warns,
    )


# ======================================================================
# Convenience functions
# ======================================================================

def edge_of_chaos_sigma_w(
    activation: str = "relu",
    sigma_b: float = 0.0,
    at_init: bool = False,
    q0: float = 1.0,
) -> float:
    """Compute the critical σ_w for a given activation.

    Returns the σ_w value at which χ_1 = 1 (edge of chaos).

    Parameters
    ----------
    activation : str
        Activation function name.
    sigma_b : float
        Bias standard deviation.
    at_init : bool
        If True, compute σ_w for criticality at the initial variance q₀,
        not at the fixed point. This is correct for initialization.
    q0 : float
        Initial input variance (only used if at_init=True).

    Returns
    -------
    float
        Critical σ_w value.
    """
    if at_init:
        # Compute σ_w such that σ_w² · E[φ'(√q₀ · z)²] = 1
        chi_map = {
            'relu': ActivationVarianceMaps.relu_chi,
            'tanh': ActivationVarianceMaps.tanh_chi,
            'gelu': ActivationVarianceMaps.gelu_chi,
            'silu': ActivationVarianceMaps.silu_chi,
            'leaky_relu': ActivationVarianceMaps.leaky_relu_chi,
            'elu': ActivationVarianceMaps.elu_chi,
            'mish': ActivationVarianceMaps.mish_chi,
            'linear': ActivationVarianceMaps.linear_chi,
        }
        chi_fn = chi_map.get(activation, chi_map.get('relu'))
        chi_act_at_q0 = chi_fn(q0)
        if chi_act_at_q0 > 1e-12:
            return 1.0 / math.sqrt(chi_act_at_q0)
        return 1.0

    mf = MeanFieldAnalyzer()
    result = mf.find_edge_of_chaos(activation, sigma_b=sigma_b)
    if isinstance(result, tuple):
        return result[0]
    return result


def compute_chi1(
    activation: str = "relu",
    sigma_w: float = 1.414,
    sigma_b: float = 0.0,
) -> float:
    """Compute χ_1 for a given activation and σ_w.

    Parameters
    ----------
    activation : str
    sigma_w : float
    sigma_b : float

    Returns
    -------
    float
        First-order susceptibility χ_1.
    """
    mf = MeanFieldAnalyzer()
    spec = ArchitectureSpec(
        depth=5, width=1000, activation=activation,
        sigma_w=sigma_w, sigma_b=sigma_b,
    )
    report = mf.analyze(spec)
    return report.chi_1


def variance_trajectory(
    activation: str = "relu",
    sigma_w: float = 1.414,
    sigma_b: float = 0.0,
    depth: int = 20,
    width: int = 1000,
    q0: float = 1.0,
    finite_width: bool = False,
) -> List[float]:
    """Compute the variance trajectory through a network.

    Parameters
    ----------
    activation : str
    sigma_w : float
    sigma_b : float
    depth : int
    width : int
    q0 : float
        Input variance.
    finite_width : bool
        Whether to apply finite-width corrections.

    Returns
    -------
    list of float
        Per-layer variance values.
    """
    mf = MeanFieldAnalyzer()
    spec = ArchitectureSpec(
        depth=depth, width=width, activation=activation,
        sigma_w=sigma_w, sigma_b=sigma_b, input_variance=q0,
    )
    report = mf.analyze(spec)

    if finite_width and width < 10000:
        return report.finite_width_corrected_variance or report.variance_trajectory
    return report.variance_trajectory


# ======================================================================
# Model creation helpers for testing
# ======================================================================

def make_mlp(
    input_dim: int, hidden_dim: int, output_dim: int,
    depth: int = 5, activation: str = "relu",
    sigma_w: float = None, sigma_b: float = 0.0,
) -> "nn.Module":
    """Create an MLP with specified initialization.

    If sigma_w is None, uses Kaiming initialization.
    """
    _require_torch()
    act_map = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
        "silu": nn.SiLU, "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU, "mish": nn.Mish,
    }
    act_cls = act_map.get(activation, nn.ReLU)

    layers = []
    dims = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        if sigma_w is not None:
            fan_in = dims[i]
            nn.init.normal_(linear.weight, 0, sigma_w / math.sqrt(fan_in))
            if sigma_b > 0:
                nn.init.normal_(linear.bias, 0, sigma_b)
            else:
                nn.init.zeros_(linear.bias)
        layers.append(linear)
        if i < len(dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


def make_transformer(
    d_model: int = 256, n_heads: int = 4, n_layers: int = 4,
    d_ff: int = 1024, activation: str = "gelu",
    sigma_w: float = None, pre_ln: bool = True,
) -> "nn.Module":
    """Create a simple Transformer encoder for testing."""
    _require_torch()

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, activation, pre_ln):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
            act_cls = act_map.get(activation, nn.GELU)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                act_cls(),
                nn.Linear(d_ff, d_model),
            )
            self.pre_ln = pre_ln

        def forward(self, x):
            if self.pre_ln:
                # Pre-LN: LN -> Attn -> Add -> LN -> FFN -> Add
                h = self.ln1(x)
                h, _ = self.attn(h, h, h, need_weights=False)
                x = x + h
                h = self.ln2(x)
                x = x + self.ffn(h)
            else:
                # Post-LN: Attn -> Add -> LN -> FFN -> Add -> LN
                h, _ = self.attn(x, x, x, need_weights=False)
                x = self.ln1(x + h)
                x = self.ln2(x + self.ffn(x))
            return x

    class SimpleTransformer(nn.Module):
        def __init__(self, d_model, n_heads, n_layers, d_ff, activation, pre_ln):
            super().__init__()
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, activation, pre_ln)
                for _ in range(n_layers)
            ])
            self.final_ln = nn.LayerNorm(d_model) if pre_ln else nn.Identity()

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.final_ln(x)

    model = SimpleTransformer(d_model, n_heads, n_layers, d_ff, activation, pre_ln)

    if sigma_w is not None:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, sigma_w / math.sqrt(fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    return model
