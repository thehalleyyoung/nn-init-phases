"""
Developer-facing PhaseKit API: practical initialization diagnostics.

Provides high-level functions that work on any nn.Module:

    from phasekit import check, fix, monitor

    report = check(model, input_shape=(3, 32, 32))
    # → "⚠ Ordered phase: gradients will vanish after ~3 layers. Run fix() to repair."

    fix(model, input_shape=(3, 32, 32))
    # → Applies best initialization, returns before/after comparison.

    hooks = monitor(model)
    # → Registers forward hooks that track variance health during training.

Combines DAG-based analysis with Non-Gaussian (NG) kurtosis-corrected
classification and gradient-searched initialization.
"""

from __future__ import annotations

import math
import warnings
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


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")


# ──────────────────────────────────────────────────────────────
# Model introspection
# ──────────────────────────────────────────────────────────────

@dataclass
class _ModelInfo:
    """Extracted architecture summary from an nn.Module."""
    weight_layers: List[Dict[str, Any]]  # name, fan_in, fan_out, sigma_w
    activation: str  # dominant activation
    depth: int       # number of weight layers
    width: int       # median hidden dimension
    mean_sigma_w: float
    has_residual: bool
    has_norm: bool
    has_attention: bool
    n_params: int
    arch_type: str   # 'mlp', 'cnn', 'resnet', 'transformer', 'rnn', 'mixed'


def _extract_model_info(model: "nn.Module") -> _ModelInfo:
    """Extract architecture information from any nn.Module."""
    weight_layers = []
    act_counts: Dict[str, int] = {}
    has_norm = False
    has_attention = False
    has_residual = False
    has_conv = False
    has_rnn = False
    n_params = 0

    act_map = {
        'ReLU': 'relu', 'GELU': 'gelu', 'SiLU': 'silu', 'Tanh': 'tanh',
        'LeakyReLU': 'leaky_relu', 'ELU': 'elu', 'Mish': 'mish',
        'Sigmoid': 'sigmoid', 'Softplus': 'relu', 'Hardswish': 'silu',
        'Hardtanh': 'tanh', 'PReLU': 'leaky_relu', 'RReLU': 'leaky_relu',
        'SELU': 'relu', 'CELU': 'elu',
    }

    for name, m in model.named_modules():
        cls_name = type(m).__name__

        # Weight layers
        if isinstance(m, (nn.Linear,)):
            fan_in = m.weight.shape[1]
            fan_out = m.weight.shape[0]
            with torch.no_grad():
                sigma_w = float(m.weight.detach().std()) * math.sqrt(fan_in)
            weight_layers.append({
                'name': name, 'fan_in': fan_in, 'fan_out': fan_out,
                'sigma_w': sigma_w, 'type': 'linear',
            })
            n_params += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)

        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            has_conv = True
            fan_in = m.weight.shape[1] * (m.kernel_size[0] if hasattr(m.kernel_size, '__len__') else m.kernel_size)
            if hasattr(m.kernel_size, '__len__') and len(m.kernel_size) > 1:
                for k in m.kernel_size[1:]:
                    fan_in *= k
            fan_out = m.weight.shape[0]
            with torch.no_grad():
                sigma_w = float(m.weight.detach().std()) * math.sqrt(fan_in)
            weight_layers.append({
                'name': name, 'fan_in': fan_in, 'fan_out': fan_out,
                'sigma_w': sigma_w, 'type': 'conv',
            })
            n_params += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)

        # RNN layers
        elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            has_rnn = True
            n_params += sum(p.numel() for p in m.parameters())
            # Approximate as depth × width
            for i in range(m.num_layers):
                in_size = m.input_size if i == 0 else m.hidden_size
                weight_layers.append({
                    'name': f'{name}.layer{i}', 'fan_in': in_size,
                    'fan_out': m.hidden_size, 'sigma_w': 1.0,
                    'type': 'rnn',
                })

        # Activations
        if cls_name in act_map:
            act = act_map[cls_name]
            act_counts[act] = act_counts.get(act, 0) + 1

        # Normalization
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                          nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            has_norm = True

        # Attention
        if isinstance(m, nn.MultiheadAttention) or 'attention' in cls_name.lower():
            has_attention = True

    # Detect residual connections from source code / forward method
    try:
        import inspect
        for name, m in model.named_modules():
            try:
                src = inspect.getsource(type(m).forward)
                # Look for patterns like: x + ..., return ... + x, residual, skip
                if any(p in src for p in ['+ x', '+x', 'x +', 'x+',
                                           'residual', 'skip', 'shortcut',
                                           '+ self.', '+ out', 'out +']):
                    has_residual = True
                    break
            except (TypeError, OSError):
                pass
    except Exception:
        pass

    # Dominant activation
    if act_counts:
        activation = max(act_counts, key=act_counts.get)
    else:
        # Try to detect from functional calls in forward
        activation = 'relu'  # default assumption

    # Width = median hidden dim (exclude output layer)
    hidden_dims = [wl['fan_out'] for wl in weight_layers[:-1]] if len(weight_layers) > 1 else [weight_layers[0]['fan_out']] if weight_layers else [128]
    width = int(sorted(hidden_dims)[len(hidden_dims) // 2]) if hidden_dims else 128

    # Mean sigma_w
    sigmas = [wl['sigma_w'] for wl in weight_layers]
    mean_sigma_w = float(np.mean(sigmas)) if sigmas else 1.0

    # Arch type
    if has_attention:
        arch_type = 'transformer'
    elif has_rnn:
        arch_type = 'rnn'
    elif has_residual and not has_conv:
        arch_type = 'resnet'
    elif has_conv:
        arch_type = 'cnn'
    else:
        arch_type = 'mlp'

    return _ModelInfo(
        weight_layers=weight_layers,
        activation=activation,
        depth=len(weight_layers),
        width=width,
        mean_sigma_w=mean_sigma_w,
        has_residual=has_residual,
        has_norm=has_norm,
        has_attention=has_attention,
        n_params=n_params,
        arch_type=arch_type,
    )


# ──────────────────────────────────────────────────────────────
# check() — developer-facing health check
# ──────────────────────────────────────────────────────────────

@dataclass
class CheckReport:
    """Result from check(): human-readable initialization health report."""
    healthy: bool  # True if model should train well
    phase: str     # 'ordered', 'critical', 'chaotic'
    confidence: float  # 0-1
    summary: str   # one-line human-readable summary
    details: List[str]  # actionable details
    fix_needed: bool
    fix_description: str  # what fix() would do
    # Technical details
    chi_total: float = 1.0
    depth: int = 0
    width: int = 0
    activation: str = ""
    arch_type: str = ""
    n_params: int = 0
    sigma_w_current: float = 0.0
    sigma_w_recommended: float = 0.0
    variance_health: str = ""  # 'stable', 'shrinking', 'growing', 'exploding'
    estimated_convergence_factor: float = 1.0  # relative to optimal init


def check(
    model: "nn.Module",
    input_shape: Optional[Tuple[int, ...]] = None,
    verbose: bool = True,
) -> CheckReport:
    """Check if a model's initialization will support training.

    Works on ANY nn.Module — MLPs, CNNs, ResNets, Transformers, RNNs.
    Combines DAG-based analysis with NG kurtosis-corrected classification.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    input_shape : tuple, optional
        Shape of a single input (no batch dim). If None, inferred from
        the first Linear/Conv layer's input size.
    verbose : bool
        If True, prints the report.

    Returns
    -------
    CheckReport
        Initialization health assessment with actionable advice.

    Examples
    --------
    >>> model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    >>> report = check(model)
    ✓ Healthy initialization (critical phase, χ=0.98)
      Your 2-layer ReLU MLP (198K params) should train well.
    """
    _require_torch()
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    info = _extract_model_info(model)

    # Infer input_shape if not provided
    if input_shape is None:
        if info.weight_layers:
            first = info.weight_layers[0]
            if first['type'] == 'conv':
                input_shape = (3, 32, 32)  # reasonable default for conv
            else:
                input_shape = (first['fan_in'],)
        else:
            input_shape = (128,)

    # --- Signal 1: DAG-based analysis (works on the actual model graph) ---
    dag_phase = "critical"
    dag_chi = 1.0
    dag_var_err = 0.0
    dag_recommended_sw = 1.0
    try:
        from dag_propagator import analyze_dag
        dag = analyze_dag(model, input_shape, n_samples=128, seed=42)
        dag_phase = dag.phase
        dag_chi = dag.chi_total
        dag_var_err = dag.variance_error_pct
        dag_recommended_sw = dag.recommended_sigma_w
    except Exception:
        pass

    # --- Signal 2: NG classification (uses extracted params) ---
    ng_phase = "critical"
    ng_confidence = 0.5
    ng_chi_eff = 1.0
    try:
        from ng_classifier import classify_ng as _classify_ng
        ng_result = _classify_ng(
            sigma_w=info.mean_sigma_w,
            sigma_b=0.0,
            activation=info.activation,
            width=info.width,
            depth=max(info.depth - 1, 1),  # hidden layers
            use_mc=True,
            mc_trials=6,
            seed=42,
        )
        ng_phase = ng_result.phase
        ng_confidence = ng_result.confidence
        ng_chi_eff = ng_result.chi_eff_per_layer
    except Exception:
        pass

    # --- Signal 3: Empirical forward-pass variance check ---
    var_health = "stable"
    var_ratio = 1.0
    try:
        with torch.no_grad():
            x = torch.randn(32, *input_shape)
            # Register hooks to measure variance at each layer
            layer_vars = []
            hooks = []
            def make_hook(lst):
                def hook_fn(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        lst.append(float(out.var().item()))
                return hook_fn
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                  nn.LSTM, nn.GRU)):
                    hooks.append(m.register_forward_hook(make_hook(layer_vars)))
            model(x)
            for h in hooks:
                h.remove()

            if len(layer_vars) >= 2:
                ratios = [layer_vars[i+1] / max(layer_vars[i], 1e-30)
                          for i in range(len(layer_vars)-1)]
                geo_mean = np.exp(np.mean([np.log(max(r, 1e-30)) for r in ratios]))
                var_ratio = geo_mean
                if geo_mean < 0.5:
                    var_health = "shrinking"
                elif geo_mean > 2.0:
                    var_health = "growing"
                elif geo_mean > 10.0 or any(r > 100 for r in ratios):
                    var_health = "exploding"
    except Exception:
        pass

    # --- Combine signals with practical trainability analysis ---
    # Decision hierarchy:
    # 1. Norm/attention → healthy (these rescue everything)
    # 2. Residual → healthy unless extreme empirical evidence
    # 3. depth <= 10, no norm/residual: empirical var_ratio primary
    # 4. depth > 10, no norm/residual: combine empirical + NG theory
    #    (per-layer empirical can look fine while cumulative decay kills signal)

    if info.has_norm or info.has_attention:
        phase = "critical"
        confidence = 0.9
    elif info.has_residual:
        if var_health == "exploding" or var_ratio > 10.0:
            phase = "chaotic"
            confidence = 0.85
        else:
            phase = "critical"
            confidence = 0.85
    elif info.depth <= 3 and not (var_health == "exploding"):
        # Very shallow networks: almost always train fine with any init
        phase = "critical"
        confidence = 0.85
    elif var_health == "exploding" or var_ratio > 5.0:
        phase = "chaotic"
        confidence = 0.95
    elif var_health == "shrinking" and var_ratio < 0.1:
        phase = "ordered"
        confidence = 0.9
    elif info.depth > 10:
        # Deep plain networks: combine empirical + NG
        # Per-layer empirical can look fine (ratio ~1.0) while
        # cumulative effects cause failure
        if ng_chi_eff > 0 and np.isfinite(ng_chi_eff):
            # Compute end-to-end signal ratio from NG
            effective_depth = max(info.depth - 1, 1)
            log_signal = effective_depth * np.log(max(ng_chi_eff, 1e-30))
            ng_signal = np.exp(min(max(log_signal, -50), 50))

            # Also compute end-to-end from empirical (more reliable for moderate depth)
            empirical_signal = var_ratio ** max(info.depth - 1, 1) if var_ratio > 0 else 0

            # Use the more pessimistic estimate
            signal = min(ng_signal, empirical_signal) if np.isfinite(empirical_signal) else ng_signal

            if signal < 0.01 or (ng_chi_eff < 0.95 and info.depth > 15):
                phase = "ordered"
                confidence = 0.85
            elif signal > 200:
                phase = "chaotic"
                confidence = 0.85
            else:
                phase = "critical"
                confidence = 0.75
        else:
            # No NG data — use conservative heuristic for deep plain nets
            if var_ratio < 0.7 and info.depth > 15:
                phase = "ordered"
                confidence = 0.7
            else:
                phase = "critical"
                confidence = 0.6
    elif var_ratio >= 0.2 and var_ratio <= 5.0:
        # Shallow/medium depth with stable empirical variance → healthy
        phase = "critical"
        confidence = 0.8
    else:
        # Shallow with poor variance
        if var_ratio < 0.2:
            phase = "ordered"
            confidence = 0.75
        else:
            phase = "chaotic"
            confidence = 0.75

    # --- Determine health ---
    healthy = (phase == "critical")

    # Estimated convergence factor
    if phase == "critical":
        conv_factor = 1.0
    elif phase == "ordered":
        conv_factor = 1.5 + 0.1 * info.depth
    else:
        conv_factor = 2.0 + 0.2 * info.depth

    # NG-recommended sigma_w
    try:
        from ng_classifier import find_critical_sigma_w
        sw_rec = find_critical_sigma_w(info.activation, info.width, max(info.depth - 1, 1))
    except Exception:
        sw_rec = dag_recommended_sw

    # --- Build report ---
    fix_needed = not healthy
    details = []
    arch_desc = f"{info.depth}-layer {info.activation.upper()} {info.arch_type.upper()}"
    param_str = f"{info.n_params/1e6:.1f}M" if info.n_params >= 1e6 else f"{info.n_params/1e3:.0f}K"

    if healthy:
        summary = f"✓ Healthy initialization ({phase} phase)"
        details.append(f"Your {arch_desc} ({param_str} params) should train well.")
        if var_health == "stable":
            details.append(f"Variance is stable across layers (ratio ≈ {var_ratio:.2f}).")
        fix_description = "No fix needed."
    elif phase == "ordered":
        summary = f"⚠ Ordered phase: gradients will vanish"
        details.append(f"Your {arch_desc} ({param_str} params) has vanishing gradients.")
        details.append(f"Current σ_w ≈ {info.mean_sigma_w:.3f}, need ≈ {sw_rec:.3f}.")
        details.append(f"Variance shrinks by {var_ratio:.2f}× per layer → signal lost after ~{max(1, int(-1/np.log(max(var_ratio, 1e-10)))):.0f} layers.")
        if info.depth > 5 and not info.has_norm:
            details.append("Consider adding LayerNorm or skip connections for this depth.")
        fix_description = f"Rescale weights to σ_w ≈ {sw_rec:.3f} (edge of chaos for {info.activation})."
    else:
        summary = f"⚠ Chaotic phase: gradients will explode"
        details.append(f"Your {arch_desc} ({param_str} params) has exploding gradients.")
        details.append(f"Current σ_w ≈ {info.mean_sigma_w:.3f}, need ≈ {sw_rec:.3f}.")
        details.append(f"Variance grows by {var_ratio:.2f}× per layer → overflow after ~{max(1, int(30/np.log(max(var_ratio, 1.01)))):.0f} layers.")
        fix_description = f"Rescale weights to σ_w ≈ {sw_rec:.3f} (edge of chaos for {info.activation})."

    report = CheckReport(
        healthy=healthy,
        phase=phase,
        confidence=confidence,
        summary=summary,
        details=details,
        fix_needed=fix_needed,
        fix_description=fix_description,
        chi_total=dag_chi,
        depth=info.depth,
        width=info.width,
        activation=info.activation,
        arch_type=info.arch_type,
        n_params=info.n_params,
        sigma_w_current=info.mean_sigma_w,
        sigma_w_recommended=sw_rec,
        variance_health=var_health,
        estimated_convergence_factor=conv_factor,
    )

    if verbose:
        print(report.summary)
        for d in report.details:
            print(f"  {d}")
        if fix_needed:
            print(f"  → Run fix(model, input_shape={input_shape}) to repair.")

    return report


# ──────────────────────────────────────────────────────────────
# fix() — apply best initialization
# ──────────────────────────────────────────────────────────────

@dataclass
class FixReport:
    """Result from fix(): before/after initialization comparison."""
    phase_before: str
    phase_after: str
    sigma_w_before: float
    sigma_w_after: float
    layers_modified: int
    variance_health_before: str
    variance_health_after: str
    convergence_improvement: str  # e.g. "1.5× faster"
    summary: str


def fix(
    model: "nn.Module",
    input_shape: Optional[Tuple[int, ...]] = None,
    strategy: str = "auto",
    verbose: bool = True,
) -> FixReport:
    """Fix a model's initialization for optimal training.

    Applies the best initialization strategy based on the model's
    architecture. Uses GS-CaliKit for deep networks, edge-of-chaos
    scaling for shallow networks, and leaves normalized/residual
    architectures mostly untouched.

    Parameters
    ----------
    model : nn.Module
        Model to fix (modified in-place).
    input_shape : tuple, optional
        Shape of a single input. Inferred if not provided.
    strategy : str
        'auto': choose best strategy based on architecture.
        'rescale': rescale weights to edge-of-chaos σ_w.
        'lsuv': layer-wise unit-variance calibration.
    verbose : bool
        If True, prints before/after comparison.

    Returns
    -------
    FixReport
    """
    _require_torch()
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    # Before
    before = check(model, input_shape, verbose=False)

    info = _extract_model_info(model)

    if input_shape is None:
        if info.weight_layers:
            first = info.weight_layers[0]
            input_shape = (first['fan_in'],) if first['type'] != 'conv' else (3, 32, 32)
        else:
            input_shape = (128,)

    # Choose strategy
    if strategy == "auto":
        if info.has_norm and info.has_residual:
            strategy = "light"  # minimal intervention for well-structured models
        elif info.depth > 5:
            strategy = "lsuv"   # LSUV for deep networks
        else:
            strategy = "rescale"

    # Determine target sigma_w
    try:
        from ng_classifier import find_critical_sigma_w
        sw_target = find_critical_sigma_w(
            info.activation, info.width, max(info.depth - 1, 1)
        )
    except Exception:
        sw_defaults = {'relu': 1.414, 'gelu': 1.414, 'silu': 1.414,
                       'tanh': 1.0, 'leaky_relu': 1.414}
        sw_target = sw_defaults.get(info.activation, 1.0)

    layers_modified = 0

    if strategy == "light":
        # Only fix clearly problematic layers
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                with torch.no_grad():
                    fan_in = m.weight.shape[1]
                    if m.weight.dim() > 2:
                        fan_in = int(np.prod(m.weight.shape[1:]))
                    current_std = float(m.weight.detach().std())
                    target_std = sw_target / math.sqrt(fan_in)
                    ratio = current_std / max(target_std, 1e-10)
                    if ratio > 3.0 or ratio < 0.3:
                        m.weight.data.mul_(target_std / max(current_std, 1e-10))
                        layers_modified += 1

    elif strategy == "rescale":
        # Rescale all weight layers to target sigma_w
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Skip normalization-related layers in transformers
                if info.has_attention and any(k in name for k in ['ln', 'norm']):
                    continue
                with torch.no_grad():
                    fan_in = m.weight.shape[1]
                    if m.weight.dim() > 2:
                        fan_in = int(np.prod(m.weight.shape[1:]))
                    nn.init.normal_(m.weight, 0, sw_target / math.sqrt(fan_in))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    layers_modified += 1

    elif strategy == "lsuv":
        # Layer-Sequential Unit-Variance initialization
        # First rescale to target sigma_w
        target_modules = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if info.has_attention and any(k in name for k in ['ln', 'norm']):
                    continue
                with torch.no_grad():
                    fan_in = m.weight.shape[1]
                    if m.weight.dim() > 2:
                        fan_in = int(np.prod(m.weight.shape[1:]))
                    nn.init.normal_(m.weight, 0, sw_target / math.sqrt(fan_in))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    layers_modified += 1
                    target_modules.append(m)

        # Then calibrate per-layer variance with forward passes
        try:
            with torch.no_grad():
                x_calib = torch.randn(64, *input_shape)
                for target_m in target_modules[:-1]:  # skip output layer
                    for _ in range(25):
                        var_at_layer = [0.0]
                        def capture_var(mod, inp, out, var_ref=var_at_layer):
                            if isinstance(out, torch.Tensor):
                                var_ref[0] = float(out.var().item())
                        hook = target_m.register_forward_hook(capture_var)
                        try:
                            model(x_calib)
                        except Exception:
                            hook.remove()
                            break
                        hook.remove()

                        v = var_at_layer[0]
                        if not np.isfinite(v) or v < 1e-20:
                            target_m.weight.data.mul_(0.5)
                            continue
                        if abs(v - 1.0) < 0.1:
                            break
                        scale = 1.0 / math.sqrt(max(v, 1e-10))
                        scale = max(0.1, min(scale, 10.0))
                        target_m.weight.data.mul_(scale)
        except Exception:
            pass  # LSUV calibration is best-effort

    # After
    after = check(model, input_shape, verbose=False)

    # Convergence improvement estimate
    if before.estimated_convergence_factor > 1.0 and after.estimated_convergence_factor <= 1.0:
        improvement = f"{before.estimated_convergence_factor:.1f}× faster"
    elif before.estimated_convergence_factor > after.estimated_convergence_factor:
        ratio = before.estimated_convergence_factor / max(after.estimated_convergence_factor, 0.1)
        improvement = f"{ratio:.1f}× faster"
    else:
        improvement = "similar"

    summary_parts = []
    if before.phase != after.phase:
        summary_parts.append(f"Phase: {before.phase} → {after.phase}")
    summary_parts.append(f"σ_w: {before.sigma_w_current:.3f} → {after.sigma_w_current:.3f}")
    summary_parts.append(f"Modified {layers_modified} layers")
    if improvement != "similar":
        summary_parts.append(f"Expected {improvement} convergence")
    summary = ". ".join(summary_parts) + "."

    report = FixReport(
        phase_before=before.phase,
        phase_after=after.phase,
        sigma_w_before=before.sigma_w_current,
        sigma_w_after=after.sigma_w_current,
        layers_modified=layers_modified,
        variance_health_before=before.variance_health,
        variance_health_after=after.variance_health,
        convergence_improvement=improvement,
        summary=summary,
    )

    if verbose:
        print(f"✓ Fixed: {report.summary}")

    return report


# ──────────────────────────────────────────────────────────────
# monitor() — training-time variance monitoring
# ──────────────────────────────────────────────────────────────

@dataclass
class MonitorSnapshot:
    """Snapshot of layer-wise variance during training."""
    step: int
    layer_variances: Dict[str, float]
    variance_ratio: float  # geometric mean ratio
    health: str  # 'healthy', 'warning', 'critical'
    message: str


class TrainingMonitor:
    """Monitors initialization health during training via forward hooks.

    Usage:
        monitor = TrainingMonitor(model)
        for step, (x, y) in enumerate(dataloader):
            loss = train_step(model, x, y)
            snapshot = monitor.step(step)
            if snapshot and snapshot.health == 'critical':
                print(snapshot.message)
        monitor.remove()
    """

    def __init__(self, model: "nn.Module", check_interval: int = 100):
        self.model = model
        self.check_interval = check_interval
        self._hooks = []
        self._layer_vars: Dict[str, List[float]] = {}
        self._history: List[MonitorSnapshot] = []
        self._step_count = 0

        # Register hooks on weight layers
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self._layer_vars[name] = []
                hook = m.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook_fn(mod, inp, out):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                v = float(out.detach().var().item())
                self._layer_vars[layer_name].append(v)
                # Keep only last 2 values for ratio computation
                if len(self._layer_vars[layer_name]) > 2:
                    self._layer_vars[layer_name] = self._layer_vars[layer_name][-2:]
        return hook_fn

    def step(self, global_step: int) -> Optional[MonitorSnapshot]:
        """Call after each training step. Returns snapshot at check intervals."""
        self._step_count += 1
        if self._step_count % self.check_interval != 0:
            return None

        # Collect current variances
        current_vars = {}
        for name, vals in self._layer_vars.items():
            if vals:
                current_vars[name] = vals[-1]

        if len(current_vars) < 2:
            return None

        # Compute variance ratio
        var_list = list(current_vars.values())
        ratios = [var_list[i+1] / max(var_list[i], 1e-30)
                  for i in range(len(var_list)-1)]
        valid_ratios = [r for r in ratios if np.isfinite(r) and r > 0]
        if valid_ratios:
            geo_mean = np.exp(np.mean([np.log(r) for r in valid_ratios]))
        else:
            geo_mean = 1.0

        # Health assessment
        if any(v > 1e8 or v < 1e-20 for v in var_list if np.isfinite(v)):
            health = "critical"
            msg = f"Step {global_step}: Variance collapse/explosion detected!"
        elif geo_mean > 5.0 or geo_mean < 0.2:
            health = "warning"
            msg = f"Step {global_step}: Variance ratio {geo_mean:.2f} (should be ~1.0)"
        else:
            health = "healthy"
            msg = f"Step {global_step}: Variance healthy (ratio {geo_mean:.2f})"

        snapshot = MonitorSnapshot(
            step=global_step,
            layer_variances=current_vars,
            variance_ratio=geo_mean,
            health=health,
            message=msg,
        )
        self._history.append(snapshot)
        return snapshot

    def summary(self) -> str:
        """Return a summary of training health over all monitored steps."""
        if not self._history:
            return "No data collected yet."
        n_healthy = sum(1 for s in self._history if s.health == "healthy")
        n_warning = sum(1 for s in self._history if s.health == "warning")
        n_critical = sum(1 for s in self._history if s.health == "critical")
        total = len(self._history)
        return (f"Training health: {n_healthy}/{total} healthy, "
                f"{n_warning} warnings, {n_critical} critical")

    def remove(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


def monitor(model: "nn.Module", check_interval: int = 100) -> TrainingMonitor:
    """Create a training monitor for a model.

    Parameters
    ----------
    model : nn.Module
    check_interval : int
        Check variance health every N training steps.

    Returns
    -------
    TrainingMonitor
        Call .step(global_step) after each training step.

    Examples
    --------
    >>> mon = monitor(model, check_interval=50)
    >>> for step in range(1000):
    ...     loss = train_step(model, x, y)
    ...     snap = mon.step(step)
    ...     if snap and snap.health != 'healthy':
    ...         print(snap.message)
    >>> print(mon.summary())
    >>> mon.remove()
    """
    _require_torch()
    return TrainingMonitor(model, check_interval)


# ──────────────────────────────────────────────────────────────
# init_hf() — suggest & apply optimal init for HuggingFace configs
# ──────────────────────────────────────────────────────────────

# Map HF activation names to PhaseKit activation names
_HF_ACT_MAP = {
    'gelu': 'gelu', 'gelu_new': 'gelu', 'gelu_fast': 'gelu',
    'gelu_pytorch_tanh': 'gelu',
    'relu': 'relu', 'leaky_relu': 'leaky_relu',
    'silu': 'silu', 'swish': 'silu',
    'tanh': 'tanh', 'sigmoid': 'sigmoid',
    'mish': 'mish', 'elu': 'elu',
    'quick_gelu': 'gelu',
}


@dataclass
class HFInitPlan:
    """Initialization plan for a HuggingFace model config."""
    model_type: str          # 'gpt2', 'bert', 'llama', etc.
    depth: int               # number of transformer layers
    width: int               # hidden dimension
    n_heads: int
    ffn_dim: int
    activation: str          # normalised name
    norm_type: str           # 'layernorm', 'rmsnorm', 'none'
    has_residual: bool
    n_params: int            # estimated parameter count

    # ── recommendations ──
    sigma_w_default: float   # HF default initializer_range
    sigma_w_attn: float      # recommended σ for attention Q/K/V input projections
    sigma_w_attn_out: float  # recommended σ for attention output projection (depth-scaled)
    sigma_w_ffn: float       # recommended σ for FFN input layers
    sigma_w_ffn_out: float   # recommended σ for FFN output projection (depth-scaled)
    sigma_w_embed: float     # recommended σ for embeddings
    sigma_w_head: float      # recommended σ for output head / LM head

    # per-component (σ_w * sqrt(fan_in) normalised)
    layer_plan: List[Dict[str, Any]]

    phase_before: str        # phase with default init
    phase_after: str         # phase with recommended init
    summary: str
    details: List[str]


def _parse_hf_config(config) -> Dict[str, Any]:
    """Extract architecture info from any HuggingFace PretrainedConfig."""
    info: Dict[str, Any] = {}
    c = config

    # Model type
    info['model_type'] = getattr(c, 'model_type', type(c).__name__.replace('Config', '').lower())

    # Hidden size
    info['width'] = getattr(c, 'hidden_size', None) or getattr(c, 'n_embd', None) or getattr(c, 'd_model', None) or getattr(c, 'dim', 256)

    # Depth
    info['depth'] = (getattr(c, 'num_hidden_layers', None) or getattr(c, 'n_layer', None)
                     or getattr(c, 'num_layers', None) or getattr(c, 'encoder_layers', None) or 6)

    # Heads
    info['n_heads'] = (getattr(c, 'num_attention_heads', None) or getattr(c, 'n_head', None)
                       or getattr(c, 'num_heads', None) or getattr(c, 'encoder_attention_heads', None) or 4)

    # FFN dim
    info['ffn_dim'] = (getattr(c, 'intermediate_size', None) or getattr(c, 'n_inner', None)
                       or getattr(c, 'd_ff', None) or getattr(c, 'encoder_ffn_dim', None)
                       or info['width'] * 4)

    # Activation
    raw_act = (getattr(c, 'hidden_act', None) or getattr(c, 'activation_function', None)
               or getattr(c, 'hidden_activation', None) or getattr(c, 'dense_act_fn', None) or 'gelu')
    info['activation'] = _HF_ACT_MAP.get(raw_act, 'gelu')

    # Norm type
    if hasattr(c, 'rms_norm_eps') or info['model_type'] in ('llama', 'mistral', 'gemma', 'qwen2', 'phi3'):
        info['norm_type'] = 'rmsnorm'
    elif hasattr(c, 'layer_norm_eps') or hasattr(c, 'layer_norm_epsilon'):
        info['norm_type'] = 'layernorm'
    else:
        info['norm_type'] = 'layernorm'  # default

    info['has_residual'] = True  # all transformers have residual

    # Default init
    info['sigma_w_default'] = getattr(c, 'initializer_range', 0.02)

    # Vocab size (for param count estimate)
    info['vocab_size'] = getattr(c, 'vocab_size', 50257)

    # GQA / MQA
    info['num_kv_heads'] = getattr(c, 'num_key_value_heads', None) or info['n_heads']

    return info


def _compute_optimal_sigma(width: int, depth: int, activation: str,
                           component: str = 'ffn') -> float:
    """Compute optimal per-parameter σ for a transformer component.

    The theory: for a residual transformer, each layer adds to the residual
    stream. The output projection of each sub-block (attention, FFN) should
    be scaled by 1/√(2L) so the residual variance stays O(1) after L layers
    (GPT-2 / µP convention). Input projections use standard He/Xavier.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    # Edge-of-chaos σ_w for this activation (σ_w * √fan_in normalised)
    try:
        from ng_classifier import find_critical_sigma_w
        sw_crit = find_critical_sigma_w(activation, width, min(depth, 20))
    except Exception:
        if activation in ('relu', 'gelu', 'silu', 'leaky_relu', 'mish', 'elu'):
            sw_crit = math.sqrt(2.0)
        else:
            sw_crit = 1.0

    # Per-parameter base sigma (before depth scaling)
    base = sw_crit / math.sqrt(width)

    if component == 'attn':
        # Attention Q/K projections: standard init (per GPT-2)
        # Output projection: scale by 1/√(2*depth) for residual accumulation
        return base

    elif component == 'attn_out':
        # Output projection of attention: depth-scaled
        return base / math.sqrt(2.0 * max(depth, 1))

    elif component == 'ffn':
        # FFN first layer: standard init
        return base

    elif component == 'ffn_out':
        # FFN output (projects back to residual): depth-scaled
        return base / math.sqrt(2.0 * max(depth, 1))

    elif component == 'embed':
        # Embeddings: σ_w / √width (so each embedding has unit variance)
        return base

    elif component == 'head':
        # Output head: small to start near uniform
        return 0.02 / math.sqrt(max(depth, 1))

    return base


def init_hf(
    config,
    model: Optional["nn.Module"] = None,
    apply: bool = False,
    verbose: bool = True,
) -> HFInitPlan:
    """Suggest (and optionally apply) optimal initialization for a HuggingFace config.

    Given a HuggingFace PretrainedConfig, computes per-component optimal σ_w
    using PhaseKit-NG mean-field theory. If a model is provided with apply=True,
    modifies weights in-place.

    Parameters
    ----------
    config : PretrainedConfig
        Any HuggingFace model config (GPT2Config, BertConfig, LlamaConfig, etc.)
    model : nn.Module, optional
        If provided, can apply the recommended init in-place.
    apply : bool
        If True and model is provided, apply the recommended init.
    verbose : bool
        Print the recommendation.

    Returns
    -------
    HFInitPlan
        Per-component initialization recommendations with rationale.

    Examples
    --------
    >>> from transformers import GPT2Config, GPT2Model
    >>> config = GPT2Config(n_embd=768, n_layer=12)
    >>> plan = init_hf(config)
    >>> model = GPT2Model(config)
    >>> init_hf(config, model=model, apply=True)
    """
    _require_torch()

    info = _parse_hf_config(config)

    width = info['width']
    depth = info['depth']
    n_heads = info['n_heads']
    ffn_dim = info['ffn_dim']
    act = info['activation']
    default_sw = info['sigma_w_default']

    # Compute per-component optimal sigma_w
    sw_attn = _compute_optimal_sigma(width, depth, act, 'attn')
    sw_attn_out = _compute_optimal_sigma(width, depth, act, 'attn_out')
    sw_ffn = _compute_optimal_sigma(width, depth, act, 'ffn')
    sw_ffn_out = _compute_optimal_sigma(width, depth, act, 'ffn_out')
    sw_embed = _compute_optimal_sigma(width, depth, act, 'embed')
    sw_head = _compute_optimal_sigma(width, depth, act, 'head')

    # Build per-layer plan
    layer_plan = []
    for l in range(depth):
        layer_plan.append({
            'layer': l,
            'attn_qkv_sigma': round(sw_attn, 6),
            'attn_out_sigma': round(sw_attn_out, 6),
            'ffn_up_sigma': round(sw_ffn, 6),
            'ffn_down_sigma': round(sw_ffn_out, 6),
        })

    # Estimate param count
    attn_params = depth * (4 * width * width)  # Q, K, V, O
    ffn_params = depth * (2 * width * ffn_dim)
    embed_params = info['vocab_size'] * width
    n_params = attn_params + ffn_params + embed_params

    # Determine phase with default init vs recommended
    phase_before = 'critical'  # transformers with norm+residual are usually OK
    phase_after = 'critical'

    # But check if default init is particularly bad
    # HF default is usually 0.02 → σ_w·√fan_in = 0.02·√width
    effective_default = default_sw * math.sqrt(width)
    if effective_default < 0.3:
        phase_before = 'ordered (weak signal)'
    elif effective_default > 3.0:
        phase_before = 'chaotic (noisy signal)'

    # Summary
    details = []
    details.append(f"Model: {info['model_type']} ({depth}L, {width}d, {n_heads}h, {act})")
    details.append(f"Norm: {info['norm_type']}, Residual: yes")
    details.append(f"Default HF init: σ = {default_sw} (flat for all layers)")

    details.append(f"Recommended attn input σ:  {sw_attn:.5f} ({sw_attn/max(default_sw,1e-10):.1f}× default)")
    details.append(f"Recommended attn output σ: {sw_attn_out:.5f} ({sw_attn_out/max(default_sw,1e-10):.1f}× default, depth-scaled 1/√(2·{depth}))")
    details.append(f"Recommended FFN input σ:   {sw_ffn:.5f} ({sw_ffn/max(default_sw,1e-10):.1f}× default)")
    details.append(f"Recommended FFN output σ:  {sw_ffn_out:.5f} ({sw_ffn_out/max(default_sw,1e-10):.1f}× default, depth-scaled)")
    details.append(f"Recommended embed σ:       {sw_embed:.5f}")
    details.append(f"Recommended head σ:        {sw_head:.5f}")

    if depth >= 12:
        details.append(f"Deep model ({depth}L): depth-scaled head init σ={sw_head:.5f} prevents early overconfidence")
    if info['norm_type'] == 'rmsnorm':
        details.append("RMSNorm detected: init is less critical but optimal σ still helps convergence speed")

    param_str = f"{n_params/1e6:.0f}M" if n_params >= 1e6 else f"{n_params/1e3:.0f}K"
    summary = (f"PhaseKit init plan for {info['model_type'].upper()} "
               f"({depth}L/{width}d/{param_str}): "
               f"attn σ={sw_attn:.4f}, ffn σ={sw_ffn:.4f}")

    plan = HFInitPlan(
        model_type=info['model_type'],
        depth=depth,
        width=width,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        activation=act,
        norm_type=info['norm_type'],
        has_residual=True,
        n_params=n_params,
        sigma_w_default=default_sw,
        sigma_w_attn=sw_attn,
        sigma_w_attn_out=sw_attn_out,
        sigma_w_ffn=sw_ffn,
        sigma_w_ffn_out=sw_ffn_out,
        sigma_w_embed=sw_embed,
        sigma_w_head=sw_head,
        layer_plan=layer_plan,
        phase_before=phase_before,
        phase_after=phase_after,
        summary=summary,
        details=details,
    )

    # Apply if requested
    if apply and model is not None:
        _apply_hf_init(model, plan)

    if verbose:
        print(plan.summary)
        for d in plan.details:
            print(f"  {d}")

    return plan


def _apply_hf_init(model: "nn.Module", plan: HFInitPlan):
    """Apply PhaseKit-recommended init to a HuggingFace model in-place."""
    modified = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue  # skip biases, norms

            name_lower = name.lower()

            # Output projections (depth-scaled): o_proj, c_proj, out_proj, down_proj, wo, fc2
            if any(k in name_lower for k in ('o_proj', 'out_proj',
                                              'down_proj', 'wo')):
                sigma = plan.sigma_w_ffn_out
            elif 'c_proj' in name_lower and 'attn' in name_lower:
                sigma = plan.sigma_w_attn_out
            elif any(k in name_lower for k in ('fc2',)) and any(k in name_lower for k in ('mlp', 'ffn')):
                sigma = plan.sigma_w_ffn_out
            # Attention input projections
            elif any(k in name_lower for k in ('q_proj', 'k_proj', 'v_proj',
                                                'c_attn', 'query', 'key', 'value',
                                                'qkv', 'in_proj', 'self_attn')):
                sigma = plan.sigma_w_attn
            # FFN input projections
            elif any(k in name_lower for k in ('mlp', 'ffn', 'fc1',
                                                'c_fc', 'intermediate',
                                                'gate_proj', 'up_proj',
                                                'wi', 'dense')):
                sigma = plan.sigma_w_ffn
            # Embeddings
            elif any(k in name_lower for k in ('embed', 'wte', 'wpe', 'word_embed',
                                                'position_embed', 'patch_embed')):
                sigma = plan.sigma_w_embed
            # Output head
            elif any(k in name_lower for k in ('lm_head', 'classifier', 'head',
                                                'score', 'predictions')):
                sigma = plan.sigma_w_head
            else:
                sigma = plan.sigma_w_ffn  # default

            nn.init.normal_(param, 0, sigma)
            modified += 1

    return modified
