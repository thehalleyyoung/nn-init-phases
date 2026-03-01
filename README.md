# PhaseKit: Mean-Field Phase Diagrams for Neural Network Initialization

Diagnose and fix neural network initialization in **one line**. PhaseKit predicts training dynamics (ordered/critical/chaotic phase) using mean-field theory — covering **MLPs, ConvNets, ResNets, Transformers**, and arbitrary computation graphs. Works on any `nn.Module` including production torchvision models.

## 30-Second Quickstart

```python
import sys; sys.path.insert(0, 'src')
from phasekit import classify, initialize, diagnose
import torch.nn as nn

# 1. Classify any model's initialization phase
model = nn.Sequential(nn.Linear(784, 256), nn.GELU(),
                      nn.Linear(256, 256), nn.GELU(),
                      nn.Linear(256, 10))
result = classify(model, input_shape=(784,))
print(result.phase, result.chi_total)  # 'ordered', 0.0037

# 2. Initialize to edge of chaos
initialize(model, input_shape=(784,))

# 3. Full diagnostic report
report = diagnose(model, input_shape=(784,))
for r in report.recommendations:
    print(r)
```

### Works on any architecture

```python
import torchvision.models as tvm
from phasekit import classify

# ResNet, VGG, MobileNet, DenseNet — all work
for name, model_fn in [('ResNet-18', tvm.resnet18),
                        ('VGG-11', tvm.vgg11),
                        ('MobileNetV2', tvm.mobilenet_v2)]:
    model = model_fn(weights=None)
    result = classify(model, input_shape=(3, 32, 32))
    print(f"{name}: {result.phase} (χ={result.chi_total:.3f})")
```

### Transformers

```python
import torch.nn as nn
from phasekit import classify

class MiniGPT(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                dim_feedforward=d_model*4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, d_model)
    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)

result = classify(MiniGPT(), input_shape=(32, 256))
print(f"Phase: {result.phase}, χ={result.chi_total:.4f}, attention={result.has_attention}")
# Phase: ordered, χ=0.1123, attention=True
```

### Init-time critical σ_w

```python
from phasekit import edge_of_chaos_sigma_w

# Fixed-point vs init-time values differ for non-ReLU activations
for act in ['relu', 'gelu', 'silu', 'tanh']:
    fp = edge_of_chaos_sigma_w(act)
    it = edge_of_chaos_sigma_w(act, at_init=True)
    print(f"{act}: fixed-point σ_w*={fp:.3f}, init-time={it:.3f}")
```

## Key Results

| Experiment | Metric | Value |
|---|---|---|
| DAG coverage | Architectures analyzed | 21/21 (100%) |
| DAG (extended) | Phase classification accuracy | 95.1% (58/61) |
| Phase classification | 3-class accuracy (210 configs) | 85.7% |
| Phase classification | Chaotic precision | 100% |
| Stochastic crossover | Width scaling exponent | 0.500 (all activations) |
| MLPs | Finite-width improvement | 3.4× at W=32 |
| Z3 verification | Properties verified | 20/20 (unsat) |
| Torchvision | Models supported | ResNet-18/50, VGG-11, MobileNetV2, DenseNet-121 |

## Features

- **Unified API**: `classify`, `initialize`, `diagnose` — three functions for all use cases
- **Arbitrary graph support**: DAG-based variance propagation via `torch.fx` for any `nn.Module`
- **Transformer support**: Self-attention variance propagation, LayerNorm reset, Pre-LN/Post-LN blocks
- **Init-time critical σ_w**: Correct initialization for non-ReLU activations (GELU, SiLU, tanh)
- **Stochastic crossover analysis**: Phase boundary width scaling Δ(N) = A·N^{-0.5}
- **9 activation functions**: ReLU, Tanh, GELU, SiLU, LeakyReLU, ELU, Mish, Sigmoid, Linear
- **Finite-width corrections**: O(1/N) + O(1/N²) using fourth and sixth moment terms
- **Z3 verification**: 20 machine-checked properties (algebraic, convergence, finite-width)

## Installation

```bash
pip install numpy scipy torch torchvision
pip install -e .
```

## Project Structure

```
src/
  phasekit.py                 # Unified API (classify, initialize, diagnose)
  dag_propagator.py           # DAG-based variance propagation engine
  mean_field_theory.py        # Core MLP mean-field analysis
  transformer_mean_field.py   # Transformer extension (attention, LayerNorm)
  compositional_mf.py         # Compositional MF engine
  stochastic_crossover.py     # Phase boundary crossover analysis
  pytorch_integration.py      # PyTorch model analysis hooks
  graph_analyzer.py           # Forward-hook variance tracer
tests/                        # Test suite (pytest)
examples/                     # Usage examples
experiments/
  run_comprehensive_validation.py   # Full validation (phase, DAG, init)
  run_stochastic_crossover.py       # Crossover width scaling
  run_z3_finite_width.py            # Z3 finite-width verification
  run_transformer_experiments.py    # Transformer validation
docs/
  theory/tool_paper.tex             # Paper (29 pages)
  reviews/                          # Peer reviews
```
