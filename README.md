# PhaseKit: Neural Network Initialization Diagnostics

Predict whether your neural network will train — **before training it**. PhaseKit classifies initialization into ordered (vanishing gradients), critical (trainable), or chaotic (exploding gradients) using mean-field theory with depth-calibrated thresholds.

Works on **any `nn.Module`**: MLPs, CNNs, ResNets, Transformers, torchvision models.

## 30-Second Quickstart

```python
import sys; sys.path.insert(0, 'src')
from phasekit import classify, initialize, diagnose
import torch.nn as nn

# Build a deep network
model = nn.Sequential(
    nn.Linear(784, 256), nn.GELU(),
    nn.Linear(256, 256), nn.GELU(),
    nn.Linear(256, 256), nn.GELU(),
    nn.Linear(256, 256), nn.GELU(),
    nn.Linear(256, 10)
)

# 1. Will it train?
result = classify(model, input_shape=(784,))
print(f"Phase: {result.phase}, χ={result.chi_total:.4f}")
# Phase: critical, χ=0.0166  (trainable!)

# 2. Fix bad initialization
initialize(model, input_shape=(784,))
# Rescales weights to edge of chaos

# 3. Full diagnostic
report = diagnose(model, input_shape=(784,))
for r in report.recommendations:
    print(r)
```

## Key Results

| Metric | Value | Validated Against |
|---|---|---|
| Phase prediction accuracy | **100%** (72/72 configs) | Real MNIST training outcomes |
| Initialization wins | **7/10** configs vs Kaiming/Xavier/LeCun | MNIST accuracy (5 seeds each) |
| PhaseKit mean accuracy | **92.9%** (vs Kaiming 93.0%) | Depth 5-10, 4 activations |
| κ₄ depth growth exponent | **0.109** (sub-linear) | Monte Carlo, 100 trials |
| Architecture coverage | **21/21** (100%) | MLPs, CNNs, ResNets, Transformers |
| Stochastic crossover | **N^(-0.5)** scaling, R²>0.99 | Monte Carlo validation |
| Z3 verified properties | **20/20** | Algebraic identities |

### Depth-Calibrated Phase Classification

The key innovation: phase boundaries depend on network depth. A shallow network (depth 3) tolerates large χ deviations; a deep network (depth 50) requires χ ≈ 1. PhaseKit uses per-layer chi thresholds `[10^(-4/L), 10^(4/L)]` calibrated against real training outcomes.

```
Depth 3:  trainable chi range [0.05, 21.5]  — almost everything trains
Depth 10: trainable chi range [0.40, 2.51]  — moderate sensitivity
Depth 50: trainable chi range [0.83, 1.20]  — must be near-critical
```

### Works on Any Architecture

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

## Installation

```bash
pip install numpy scipy torch
pip install -e .
```

## Features

- **3-function API**: `classify`, `initialize`, `diagnose`
- **Depth-calibrated thresholds**: Phase boundaries account for network depth
- **DAG variance propagation**: `torch.fx` tracing handles arbitrary graphs
- **9 activations**: ReLU, GELU, SiLU, Tanh, LeakyReLU, ELU, Mish, Sigmoid, Linear
- **Finite-width corrections**: O(1/N) + O(1/N²) with empirical κ₄ bounds
- **End-to-end validated**: Phase predictions match real training outcomes
- **CPU-only**: No GPU or training data required

## Project Structure

```
src/
  phasekit.py               # Unified API (classify, initialize, diagnose)
  dag_propagator.py         # DAG-based variance propagation
  mean_field_theory.py      # Core mean-field analysis
  transformer_mean_field.py # Transformer extension
  stochastic_crossover.py   # Phase boundary crossover analysis
experiments/
  e2e_validation.py         # End-to-end training validation
  results/                  # Experiment results (JSON)
docs/
  theory/tool_paper.tex     # Paper
  meta/critiques.json       # Peer reviews
```

## Limitations

- Transformer attention variance uses near-uniform approximation at initialization
- κ₄ (excess kurtosis) is bounded but nonzero: Gaussian closure introduces controlled bias
- Z3 verification covers algebraic identities, not system-level guarantees
- `torch.fx` tracing may fail on dynamic control flow
