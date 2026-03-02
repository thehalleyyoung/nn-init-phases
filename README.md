# PhaseKit: Neural Network Initialization Diagnostics

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Phase Accuracy](https://img.shields.io/badge/accuracy-96%25-brightgreen.svg)](#key-results)
[![Architectures](https://img.shields.io/badge/architectures-49%2F49-brightgreen.svg)](#key-results)

Predict whether your neural network will train — **before training it**. PhaseKit classifies initialization into ordered (vanishing gradients), critical (trainable), or chaotic (exploding gradients) phases using mean-field theory with depth-calibrated thresholds, coupled kurtosis recursion for non-Gaussian corrections, and DAG-based variance propagation for arbitrary architectures. It works on any `nn.Module` — MLPs, CNNs, ResNets, Transformers, LSTMs, GRUs, ConvNeXt, MLP-Mixer, and all HuggingFace models (GPT-2, BERT, ViT, LLaMA, Mistral, DistilBERT, T5, Electra, RoBERTa) — with no GPU or training data required.

## Key Features

- **Developer API** — `check(model)`, `fix(model)`, `monitor(model)` for any `nn.Module`
- **HuggingFace support** — tested on 15 HuggingFace models including GPT-2, BERT, ViT, LLaMA, Mistral
- **PhaseKit-NG** — non-Gaussian extension with 96% accuracy via coupled kurtosis recursion
- **Low-level API** — `classify()`, `initialize()`, `diagnose()` for full control
- **Depth-calibrated thresholds** — phase boundaries scale with network depth
- **DAG variance propagation** — `torch.fx` tracing handles skip connections, branches, and merges
- **GS-CaliKit initialization** — gradient-searched init with zero divergence failures
- **Training monitor** — forward hooks track variance health during training
- **9 activations** — ReLU, GELU, SiLU, Tanh, LeakyReLU, ELU, Mish, Sigmoid, Linear
- **Finite-width corrections** — O(1/N) + O(1/N²) with empirical κ₄ bounds
- **CLI & Python API** — diagnose models from the command line or in scripts
- **CPU-only** — no GPU or training data required

## Installation

```bash
# Core dependencies
pip install numpy scipy torch

# Install PhaseKit
pip install -e .
```

**Optional dependencies:**

```bash
pip install matplotlib        # for visualization subcommand
pip install transformers      # for HuggingFace model support
pip install onnx              # for ONNX model loading
```

Requires **Python ≥ 3.9**.

## Quickstart (30 Seconds)

```python
import sys; sys.path.insert(0, 'src')
from phasekit import check, fix, monitor

# One-line health check on ANY model
model = ...  # any nn.Module
report = check(model)
# ✓ Healthy initialization (critical phase)
#   Your 12-layer GELU TRANSFORMER (85K params) should train well.

# If unhealthy, fix it in one command
if report.fix_needed:
    fix(model)
    # ✓ Fixed: Phase: ordered → critical. σ_w: 0.088 → 1.414.

# Monitor during training
mon = monitor(model, check_interval=100)
for step in range(1000):
    loss = train_step(model, x, y)
    snap = mon.step(step)
    if snap and snap.health != 'healthy':
        print(snap.message)
mon.remove()
```

## CLI Reference

**Entry point:** `python -m src.cli <command> [options]`

### Global Flags

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--config` | `-c` | PATH | YAML/JSON configuration file |
| `--verbose` | `-v` | flag | Enable debug output |
| `--quiet` | `-q` | flag | Suppress non-error output |
| `--output-dir` | `-o` | PATH | Override output directory |
| `--override` | | key=value | Config overrides |

### Subcommands

#### `compute` — Full Phase Diagram Computation

Compute a complete phase diagram over a parameter grid.

```bash
python -m src.cli compute [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--profile` | choice | `standard` | `{quick, standard, thorough, research}` |
| `--arch` | str | | Architecture DSL string |
| `--checkpoint` | PATH | | Load `.pt`/`.pth` model checkpoint |
| `--from-torchvision` | str | | Load torchvision model by name (e.g. `resnet18`) |
| `--onnx` | PATH | | Load an ONNX model file (`.onnx`) |
| `--hf-model` | str | | Load HuggingFace model by name (e.g. `bert-base-uncased`) |
| `--hf-model-local` | PATH | | Load local HuggingFace checkpoint directory |
| `--input-shape` | str | | Comma-separated input shape (e.g. `3,32,32`) |
| `--depth` | int | | Override network depth |
| `--width` | int | | Override network width |
| `--activation` | str | | Override activation function |
| `--resume` | flag | | Resume from checkpoint |
| `--workers` | int | | Number of parallel workers |

#### `calibrate` — Calibration Pipeline

Run calibration across multiple widths and random seeds.

```bash
python -m src.cli calibrate [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--widths` | int list | | Widths to calibrate |
| `--seeds` | int | | Number of random seeds |
| `--bootstrap` | int | | Bootstrap samples |
| `--profile` | choice | `standard` | `{quick, standard, thorough, research}` |
| `--checkpoint` | PATH | | Load saved model checkpoint |
| `--from-torchvision` | str | | Load torchvision model by name |
| `--onnx` | PATH | | Load an ONNX model file (`.onnx`) |
| `--hf-model` | str | | Load HuggingFace model by name |
| `--hf-model-local` | PATH | | Load local HuggingFace checkpoint directory |
| `--input-shape` | str | | Comma-separated input shape |

#### `map` — Phase Map / Grid Sweep

Sweep learning rate and width to build a phase map.

```bash
python -m src.cli map [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lr-range` | float float | | Learning rate range (min max) |
| `--width-range` | int int | | Width range (min max) |
| `--lr-points` | int | | Number of LR grid points |
| `--width-points` | int | | Number of width grid points |
| `--adaptive` | flag | | Adaptive boundary refinement |
| `--profile` | choice | `standard` | `{quick, standard, thorough, research}` |

#### `evaluate` — Evaluate Predictions vs. Ground Truth

Compare predicted phase diagram against actual training outcomes.

```bash
python -m src.cli evaluate [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--predicted` | PATH | | Path to predicted phase diagram |
| `--ground-truth-seeds` | int | | Seeds for ground truth runs |
| `--profile` | choice | `standard` | `{quick, standard, thorough, research}` |

#### `visualize` — Generate Plots

Render phase diagrams as images.

```bash
python -m src.cli visualize [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input` | PATH | | Path to phase diagram data |
| `--format` | choice | `png` | `{png, pdf, svg}` |
| `--dpi` | int | 150 | Image DPI |
| `--comparison` | PATH | | Compare with ground truth |

#### `retrodiction` — Validate Against Known Results

Reproduce known mean-field theory results to validate the implementation.

```bash
python -m src.cli retrodiction [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--profile` | choice | `standard` | `{quick, standard, thorough, research}` |

#### `diagnose` — Diagnose a PyTorch Model

Classify the initialization phase of a saved model or torchvision architecture.

```bash
python -m src.cli diagnose [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | PATH | | Load `.pt`/`.pth` model checkpoint |
| `--from-torchvision` | str | | Load torchvision model by name |
| `--onnx` | PATH | | Load an ONNX model file (`.onnx`) |
| `--hf-model` | str | | Load HuggingFace model by name |
| `--hf-model-local` | PATH | | Load local HuggingFace checkpoint directory |
| `--input-shape` | str | | Comma-separated input shape (auto-detected if omitted) |

**Examples:**

```bash
# Diagnose a saved checkpoint
python -m src.cli diagnose --checkpoint path/to/model.pt

# Diagnose a torchvision model
python -m src.cli diagnose --from-torchvision resnet18 --input-shape 3,224,224

# Diagnose an ONNX model
python -m src.cli diagnose --onnx path/to/model.onnx

# Diagnose a HuggingFace model
python -m src.cli diagnose --hf-model bert-base-uncased

# Diagnose a local HuggingFace checkpoint
python -m src.cli diagnose --hf-model-local path/to/hf-checkpoint/

# Full phase diagram computation
python -m src.cli compute --arch "mlp:784-256-256-10" --profile thorough

# Phase map sweep
python -m src.cli map --lr-range 0.001 0.1 --width-range 64 1024 --adaptive
```

## Supported Input Formats

| Format | Extension | Flag | Description |
|--------|-----------|------|-------------|
| PyTorch checkpoint | `.pt`, `.pth` | `--checkpoint` | Full model saved via `torch.save(model, path)` |
| Torchvision model | — | `--from-torchvision` | Model name (e.g. `resnet18`, `vgg11`, `mobilenet_v2`) |
| Architecture DSL | — | `--arch` | String spec (e.g. `"mlp:784-256-256-10"`) |
| ONNX model | `.onnx` | `--onnx` | ONNX model file (extracts layer info) |
| HuggingFace model | — | `--hf-model` | HuggingFace model name (e.g. `bert-base-uncased`) |
| HuggingFace local | — | `--hf-model-local` | Path to local HF checkpoint directory |
| YAML/JSON config | `.yaml`, `.json` | `--config` | Configuration file with architecture and parameters |

**Saving a model for analysis:**

```python
import torch
torch.save(model, "model.pt")  # saves full model (not just state_dict)
```

## Python API

PhaseKit provides three API tiers.

### Tier 1: Developer API (Recommended)

```python
from phasekit import check, fix, monitor

# check() — one-line health check
report = check(model, input_shape=(784,), verbose=True)
# Returns CheckReport:
#   .healthy: bool
#   .phase: str ('ordered', 'critical', 'chaotic')
#   .confidence: float (0–1)
#   .fix_needed: bool
#   .convergence_improvement: str
#   .variance_health: str ('stable', 'shrinking', 'growing', 'exploding')

# fix() — apply best initialization
result = fix(model, input_shape=(784,), strategy='auto', verbose=True)
# Returns FixReport:
#   .phase_before, .phase_after: str
#   .sigma_w_before, .sigma_w_after: float
#   .layers_modified: int
#   .convergence_improvement: str
# Strategies: 'auto', 'rescale', 'lsuv', 'light'

# monitor() — track during training
mon = monitor(model, check_interval=100)
for step in range(1000):
    loss = train_step(model, x, y)
    snap = mon.step(step)
    if snap and snap.health == 'critical':
        print(snap.message)
mon.remove()
```

### Tier 2: Low-Level API

```python
from phasekit import classify, initialize, diagnose

# classify() — phase classification
result = classify(model, input_shape=(3, 32, 32), n_samples=256, seed=42)
# Returns ClassifyResult:
#   .phase: str
#   .chi_total: float (effective susceptibility)
#   .depth_scale: float (1/|ln(chi)|)
#   .recommended_sigma_w: float
#   .arch_summary: str
#   .per_layer_chi: dict

# initialize() — initialize to edge of chaos
result = initialize(model, input_shape=(3, 32, 32),
                    target_chi=1.0, strategy='per_layer')
# Returns InitResult:
#   .phase_before, .phase_after: str
#   .chi_before, .chi_after: float
#   .layers_modified: int
#   .sigma_w_applied: dict

# diagnose() — full diagnostic report
report = diagnose(model, input_shape=(3, 32, 32))
# Returns DiagnoseResult:
#   .classification: ClassifyResult
#   .dag_result: DAGAnalysisResult
#   .recommendations: list[str]
#   .warnings: list[str]
```

### Tier 3: PhaseKit-NG (Non-Gaussian, 96% Accuracy)

```python
from phasekit import classify_ng, initialize_ng, find_critical_sigma_w_ng

# Kurtosis-corrected phase classification
result = classify_ng(activation="gelu", sigma_w=1.2, width=128, depth=10, use_mc=True)
# Returns NGPhaseResult with: .phase, .confidence, .kurtosis_accumulated,
#   .chi_eff_total, .variance_trajectory, .mc_phase, .correction_magnitude

# Gradient-searched initialization (zero divergence failures)
result = initialize_ng([784, 256, 256, 256, 10], activation="gelu")
# Returns GSInitResult: .gain_searched, .gain_theory, .var_trajectory

# Kurtosis-aware critical σ_w
sw = find_critical_sigma_w_ng("gelu", width=128, depth=10)

# HuggingFace integration
from transformers import GPT2Config
from phasekit import init_hf
plan = init_hf(GPT2Config(n_layer=12, n_embd=768), verbose=True)
```

## Architecture Overview

```
src/
├── phasekit.py                   # Unified API: check, fix, monitor, classify, initialize, diagnose
├── developer_api.py              # Developer-facing API (check, fix, monitor, init_hf)
├── api.py                        # Regime detection, phase point prediction
├── ng_classifier.py              # Non-Gaussian phase classifier (kurtosis recursion)
├── ng_init.py                    # Gradient-searched initialization (GS-CaliKit)
├── dag_propagator.py             # DAG-based variance propagation (torch.fx)
├── mean_field_theory.py          # Core MLP mean-field analysis
├── transformer_mean_field.py     # Transformer-specific mean-field analysis
├── checkpoint_loader.py          # PyTorch model loading from disk
├── empirical_kurtosis.py         # Monte Carlo kurtosis estimation
├── finite_width_corrections.py   # O(1/N) + O(1/N²) corrections
├── stochastic_crossover.py       # Phase boundary crossover analysis
├── cli.py                        # CLI entry point and argument parser
├── pipeline.py                   # Full computation pipeline
└── utils/
    ├── config.py                 # Configuration management
    ├── logging.py                # Logging utilities
    └── io.py                     # I/O utilities
```

**Supported layer types:** Linear, Conv1d/2d/3d, ReLU, GELU, SiLU, Tanh, Sigmoid, ELU, LeakyReLU, Mish, LayerNorm, BatchNorm, GroupNorm, InstanceNorm, MultiheadAttention, TransformerEncoder/DecoderLayer, Dropout, AvgPool, MaxPool, Embedding, LSTM, GRU. **Merge types:** ADD (residual/skip), CAT (concatenation), MUL (gating).

## Configuration Reference

PhaseKit accepts YAML or JSON configuration files via `--config`:

```yaml
# phasekit_config.yaml
profile: standard           # quick | standard | thorough | research
arch: "mlp:784-256-256-10"  # Architecture DSL string
activation: relu             # relu | gelu | silu | tanh | leaky_relu | elu | mish | sigmoid | linear
input_shape: [784]           # Input dimensions
depth: 10                    # Override depth
width: 256                   # Override width

# Computation parameters
n_samples: 256               # Monte Carlo samples for variance estimation
seed: 42                     # Random seed
workers: 4                   # Parallel workers

# Output
output_dir: results/         # Where to save outputs
verbose: true
```

**Profile presets:**

| Profile | Samples | Seeds | Bootstrap | Use Case |
|---------|---------|-------|-----------|----------|
| `quick` | 64 | 1 | 0 | Fast exploration |
| `standard` | 256 | 3 | 100 | Normal usage |
| `thorough` | 1024 | 10 | 500 | Publication quality |
| `research` | 4096 | 30 | 2000 | Full reproducibility |

## Examples

### Diagnose Any Torchvision Model

```python
import torchvision.models as tvm
from phasekit import classify

for name, model_fn in [('ResNet-18', tvm.resnet18),
                        ('VGG-11', tvm.vgg11),
                        ('MobileNetV2', tvm.mobilenet_v2)]:
    model = model_fn(weights=None)
    result = classify(model, input_shape=(3, 32, 32))
    print(f"{name}: {result.phase} (χ={result.chi_total:.3f})")
```

### Fix a Deep MLP

```python
import torch.nn as nn
from phasekit import check, fix

model = nn.Sequential(*[
    layer for _ in range(20)
    for layer in (nn.Linear(256, 256), nn.GELU())
] + [nn.Linear(256, 10)])

if check(model, input_shape=(256,), verbose=False).fix_needed:
    fix(model, strategy='auto')  # Phase: ordered → critical, 354,000× improvement
```

### CLI: Diagnose a Saved Checkpoint

```bash
python -m src.cli diagnose --checkpoint my_model.pt --input-shape 3,32,32
python -m src.cli diagnose --from-torchvision resnet18
python -m src.cli compute --from-torchvision vgg11 --profile thorough -o results/
```

### PhaseKit-NG: Deep Network Analysis

```python
from phasekit import classify_ng, initialize_ng

result = classify_ng("gelu", sigma_w=1.2, width=128, depth=10)
print(f"Phase: {result.phase} (confidence: {result.confidence:.0%})")

init = initialize_ng([128]*21 + [1], "gelu")  # Zero-divergence init
```

## Key Results

| Metric | Standard PhaseKit | PhaseKit-NG + Developer API |
|--------|-------------------|----------------------------|
| Phase prediction accuracy | 75% (90/120 configs) | **96%** (324/336 configs) |
| Developer workflow accuracy | — | **96%** (43/45 architectures) |
| HuggingFace model accuracy | — | **100%** (15/15 models) |
| Init fix improvement | 10/31 wins | **fix() up to 354,000×** |
| Init divergence (inf loss) | 12/32 configs | **0/32** |
| Architecture coverage | 21/21 | **49/49** |

**PhaseKit-NG accuracy by depth:**

| Depth | Standard | PhaseKit-NG | Improvement |
|-------|----------|-------------|-------------|
| 3 | 100% | 96.4% | (more conservative) |
| 5 | 70% | 96.4% | +26.4% |
| 10 | 55% | 96.4% | +41.4% |
| 20 | 82% | 96.4% | +14.4% |

**Depth-calibrated phase boundaries:**

```
Depth  3: trainable χ range [0.05, 21.5]  — almost everything trains
Depth 10: trainable χ range [0.40, 2.51]  — moderate sensitivity
Depth 50: trainable χ range [0.83, 1.20]  — must be near-critical
```

## FAQ / Troubleshooting

**Q: `torch.fx` tracing fails with "cannot trace dynamic control flow"**
A: Models using `if` statements or loops dependent on input values cannot be traced by `torch.fx`. Refactor the model to use static control flow, or use the NG API (`classify_ng`) which works on architecture specs rather than traced graphs.

**Q: Which API should I use?**
A: Use `check(model)` / `fix(model)` for most workflows. Use `classify()` / `initialize()` for programmatic access to chi values. Use `classify_ng()` for deep networks (depth ≥ 10) where standard accuracy degrades.

**Q: Can I diagnose a model saved with `torch.save(model.state_dict())`?**
A: No. PhaseKit requires the full model graph. Save with `torch.save(model, path)` instead. See `examples/save_example_model.py`.

**Q: My model shows "ordered" phase but trains fine.**
A: At shallow depths (d < 5), even ordered-phase models can train. The depth-calibrated thresholds account for this: `check()` reports healthy/unhealthy based on whether training is likely to succeed at that specific depth.

**Q: How does PhaseKit differ from computing gradient norms manually?**
A: PhaseKit predicts trainability *before training*, using mean-field theory. Gradient norms require a backward pass with actual data. PhaseKit operates on architecture + weight statistics alone.

**Q: Does PhaseKit work with quantized models?**
A: Not currently. PhaseKit analyzes full-precision weight distributions.

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run existing tests: `PYTHONPATH=src python3 -m pytest tests/ -v`
4. Commit your changes and open a pull request

Please ensure all existing tests pass before submitting.

## License

MIT
