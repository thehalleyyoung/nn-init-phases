"""Load PyTorch checkpoints (.pt/.pth), torchvision, ONNX, and HuggingFace models for phase analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")


def load_checkpoint(path: str) -> "nn.Module":
    """Load a PyTorch model from a .pt/.pth checkpoint file.

    Parameters
    ----------
    path : str
        Path to a .pt or .pth file containing a saved nn.Module
        (saved via ``torch.save(model, path)``).

    Returns
    -------
    nn.Module
        The loaded model in eval mode on CPU.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    ValueError
        If the file contains a state_dict instead of a full model,
        or contains an unsupported object type.
    """
    _require_torch()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if p.suffix not in (".pt", ".pth"):
        raise ValueError(f"Expected .pt or .pth file, got: {p.suffix}")

    obj = torch.load(str(p), map_location="cpu", weights_only=False)

    # state_dict detection: plain dict of tensors
    if isinstance(obj, dict) and not isinstance(obj, nn.Module):
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            raise ValueError(
                "The checkpoint contains a state_dict (weight tensors only), "
                "not a full model. Save your model with:\n"
                "    torch.save(model, 'model.pt')  # saves full model\n"
                "Instead of:\n"
                "    torch.save(model.state_dict(), 'model.pt')  # state_dict only"
            )
        # Could be a training checkpoint dict with 'model' key
        if "model" in obj and isinstance(obj["model"], nn.Module):
            obj = obj["model"]
        elif "model_state_dict" in obj or "state_dict" in obj:
            raise ValueError(
                "The checkpoint contains a training checkpoint dict with a "
                "state_dict, not a full model. Save with:\n"
                "    torch.save(model, 'model.pt')"
            )
        else:
            raise ValueError(
                f"Checkpoint contains a dict but not an nn.Module. "
                f"Keys: {list(obj.keys())[:10]}"
            )

    if not isinstance(obj, nn.Module):
        raise ValueError(
            f"Expected nn.Module, got {type(obj).__name__}. "
            f"Save your model with: torch.save(model, 'model.pt')"
        )

    obj.eval()
    return obj


def detect_input_shape(model: "nn.Module") -> Tuple[int, ...]:
    """Detect input shape from the first layer's weight dimensions.

    Inspects the first Linear or Conv layer to infer expected input shape.

    Returns
    -------
    tuple of int
        Inferred input shape (without batch dimension).
    """
    _require_torch()

    for module in model.modules():
        if isinstance(module, nn.Linear):
            return (module.in_features,)
        if isinstance(module, nn.Conv1d):
            return (module.in_channels, 32)
        if isinstance(module, nn.Conv2d):
            return (module.in_channels, 32, 32)
        if isinstance(module, nn.Conv3d):
            return (module.in_channels, 8, 8, 8)

    raise ValueError(
        "Could not detect input shape: no Linear or Conv layer found. "
        "Specify --input-shape manually."
    )


def load_torchvision_model(name: str) -> "nn.Module":
    """Load a torchvision model by name with random weights.

    Parameters
    ----------
    name : str
        Model name, e.g. "resnet18", "vgg11", "mobilenet_v2".

    Returns
    -------
    nn.Module
        The model with random initialization in eval mode.
    """
    _require_torch()

    try:
        import torchvision.models as tvm
    except ImportError:
        raise ImportError(
            "torchvision is required for --from-torchvision. Install with:\n"
            "    pip install torchvision"
        )

    # Look up model constructor by name
    model_fn = getattr(tvm, name, None)
    if model_fn is None or not callable(model_fn):
        available = [n for n in dir(tvm) if not n.startswith("_") and callable(getattr(tvm, n, None))]
        raise ValueError(
            f"Unknown torchvision model: '{name}'. "
            f"Available models include: {', '.join(sorted(available)[:20])}..."
        )

    model = model_fn(weights=None)
    model.eval()
    return model


def load_onnx_model(path: str) -> Dict[str, Any]:
    """Extract architecture description from an ONNX model file.

    Parameters
    ----------
    path : str
        Path to a ``.onnx`` file.

    Returns
    -------
    dict
        Architecture description with keys such as ``n_layers``, ``d_model``,
        ``layer_types``, and ``initializer_shapes``.

    Raises
    ------
    FileNotFoundError
        If the ONNX file does not exist.
    ImportError
        If the ``onnx`` package is not installed.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for --onnx. Install with:\n"
            "    pip install onnx"
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    if p.suffix != ".onnx":
        raise ValueError(f"Expected .onnx file, got: {p.suffix}")

    model = onnx.load(str(p))
    onnx.checker.check_model(model)

    # Count relevant op types
    linear_ops = {"MatMul", "Gemm"}
    conv_ops = {"Conv"}
    op_counts: Dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    n_linear = sum(op_counts.get(op, 0) for op in linear_ops)
    n_conv = sum(op_counts.get(op, 0) for op in conv_ops)
    n_layers = n_linear + n_conv

    # Extract dimensions from initializer tensors
    init_shapes = []
    max_dim = 0
    for init in model.graph.initializer:
        dims = list(init.dims)
        if len(dims) >= 2:
            init_shapes.append(dims)
            max_dim = max(max_dim, max(dims))

    # Infer d_model as the most common hidden dimension
    dim_counts: Dict[int, int] = {}
    for shape in init_shapes:
        for d in shape:
            if d > 1:
                dim_counts[d] = dim_counts.get(d, 0) + 1
    d_model = max(dim_counts, key=dim_counts.get) if dim_counts else 0

    return {
        "n_layers": n_layers,
        "n_linear": n_linear,
        "n_conv": n_conv,
        "d_model": d_model,
        "op_counts": op_counts,
        "initializer_shapes": init_shapes,
        "source": "onnx",
    }


def load_hf_model(name: str, load_weights: bool = False) -> Dict[str, Any]:
    """Extract architecture description from a HuggingFace model name.

    Parameters
    ----------
    name : str
        HuggingFace model identifier (e.g. ``"bert-base-uncased"``).
    load_weights : bool
        If True, also download and return the full ``nn.Module`` via
        ``AutoModel.from_pretrained``.  Default is False (config only).

    Returns
    -------
    dict
        Architecture description with keys such as ``n_layers``, ``d_model``,
        ``d_ff``, ``activation``, and optionally ``model``.

    Raises
    ------
    ImportError
        If the ``transformers`` package is not installed.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required for --hf-model. Install with:\n"
            "    pip install transformers"
        )

    config = AutoConfig.from_pretrained(name)

    arch_info = _extract_hf_config_fields(config)
    arch_info["source"] = "huggingface"
    arch_info["model_name"] = name

    if load_weights:
        from transformers import AutoModel
        arch_info["model"] = AutoModel.from_pretrained(name).eval()

    return arch_info


def load_hf_model_local(path: str) -> Dict[str, Any]:
    """Extract architecture description from a local HuggingFace checkpoint.

    Reads ``config.json`` from the given directory without network access.

    Parameters
    ----------
    path : str
        Path to a directory containing ``config.json`` (and optionally
        model weights).

    Returns
    -------
    dict
        Architecture description dict.
    """
    p = Path(path)
    config_file = p / "config.json" if p.is_dir() else p
    if not config_file.exists():
        raise FileNotFoundError(
            f"config.json not found in {path}. "
            "Ensure the path points to a HuggingFace checkpoint directory."
        )

    with open(config_file, "r") as f:
        raw = json.load(f)

    arch_info: Dict[str, Any] = {
        "n_layers": raw.get("num_hidden_layers", raw.get("n_layer", 0)),
        "d_model": raw.get("hidden_size", raw.get("n_embd", 0)),
        "d_ff": raw.get("intermediate_size", raw.get("n_inner", 0)),
        "activation": raw.get("hidden_act", raw.get("activation_function", "unknown")),
        "source": "huggingface_local",
        "model_name": raw.get("_name_or_path", str(path)),
    }
    return arch_info


def _extract_hf_config_fields(config) -> Dict[str, Any]:
    """Pull common architecture fields from a HuggingFace config object."""
    return {
        "n_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)),
        "d_model": getattr(config, "hidden_size", getattr(config, "n_embd", 0)),
        "d_ff": getattr(config, "intermediate_size", getattr(config, "n_inner", 0)),
        "activation": getattr(config, "hidden_act", getattr(config, "activation_function", "unknown")),
    }
