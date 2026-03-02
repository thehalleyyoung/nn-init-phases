# PhaseKit API Reference

## Developer API (`phasekit.check`, `phasekit.fix`, `phasekit.monitor`)

The recommended interface for most users. Three functions cover the full developer workflow.

```python
from phasekit import check, fix, monitor
```

### `check(model, input_shape=None, verbose=True) -> CheckReport`

One-line health check for any PyTorch model. Works on MLPs, CNNs, ResNets, Transformers, LSTMs, and all HuggingFace models.

```python
report = check(model)
# ✓ Healthy initialization (critical phase)
#   Your 12-layer GELU TRANSFORMER (85K params) should train well.

print(report.healthy)        # True/False
print(report.phase)          # 'ordered', 'critical', 'chaotic'
print(report.fix_needed)     # True if fix() recommended
print(report.confidence)     # 0-1
print(report.arch_type)      # 'mlp', 'cnn', 'resnet', 'transformer', 'rnn'
print(report.variance_health) # 'stable', 'shrinking', 'growing', 'exploding'
```

**Parameters:**
- `model: nn.Module` — Any PyTorch model
- `input_shape: tuple, optional` — Single input shape (no batch dim). Auto-inferred if omitted.
- `verbose: bool = True` — Print human-readable report

**Validated on 49 architectures** (96% accuracy), including 15 HuggingFace models (100% accuracy).

### `fix(model, input_shape=None, strategy='auto', verbose=True) -> FixReport`

Fix a model's initialization for optimal training (modifies in-place).

```python
report = fix(model)
# ✓ Fixed: Phase: ordered → critical. σ_w: 0.088 → 1.414. Expected 3.5× faster convergence.

print(report.phase_before, '->', report.phase_after)
print(report.layers_modified)
print(report.convergence_improvement)  # e.g. "3.5× faster"
```

**Parameters:**
- `model: nn.Module` — Model to fix (modified in-place)
- `input_shape: tuple, optional` — Auto-inferred if omitted
- `strategy: str = 'auto'` — `'auto'`, `'rescale'`, `'lsuv'`, or `'light'`
- `verbose: bool = True`

**Strategies:**
- `auto`: Chooses best strategy based on architecture (recommended)
- `rescale`: Rescale all weights to edge-of-chaos σ_w
- `lsuv`: Layer-Sequential Unit-Variance (LSUV) calibration for deep networks
- `light`: Minimal intervention for well-structured models (norm+residual)

### `monitor(model, check_interval=100) -> TrainingMonitor`

Monitor initialization health during training via forward hooks.

```python
mon = monitor(model, check_interval=50)
for step in range(1000):
    loss = train_step(model, x, y)
    snap = mon.step(step)
    if snap and snap.health == 'critical':
        print(snap.message)
print(mon.summary())
mon.remove()
```

**TrainingMonitor methods:**
- `.step(global_step) -> Optional[MonitorSnapshot]` — Returns snapshot at check intervals
- `.summary() -> str` — Training health summary
- `.remove()` — Remove all hooks

---

## Low-Level API (`phasekit.classify`, `phasekit.initialize`, `phasekit.diagnose`)

For advanced users who need detailed phase analysis.

```python
from phasekit import classify, initialize, diagnose
from phasekit import classify_ng, initialize_ng, find_critical_sigma_w_ng
```

### `classify(model, input_shape, ...) -> ClassifyResult`

```python
result = classify(model, input_shape=(3, 32, 32))
print(result.phase)          # 'ordered', 'critical', or 'chaotic'
print(result.chi_total)      # Effective susceptibility
print(result.depth_scale)    # 1/|ln(chi)|
print(result.recommended_sigma_w)  # σ_w for edge-of-chaos
print(result.arch_summary)   # Architecture description
```

**Parameters:**
- `model: nn.Module` — Any PyTorch model
- `input_shape: tuple` — Single input shape (no batch dim)
- `n_samples: int = 256` — Batch size for empirical measurement
- `seed: int = 42`
- `seq_len: int = 128` — For attention layers
- `finite_width: bool = True` — Apply O(1/N) corrections

### `initialize(model, input_shape, ...) -> InitResult`

Initialize a model to edge of chaos (modifies in-place).

```python
result = initialize(model, input_shape=(784,), strategy='depth_aware')
print(result.phase_before, '->', result.phase_after)
print(result.layers_modified)
```

**Parameters:**
- `model: nn.Module` — Model to initialize (modified in-place)
- `input_shape: tuple`
- `target_chi: float = 1.0` — Target susceptibility
- `strategy: str = 'per_layer'` — `'per_layer'`, `'global'`, or `'depth_aware'`

### `diagnose(model, input_shape, ...) -> DiagnoseResult`

Full diagnostic report with actionable recommendations.

```python
report = diagnose(model, input_shape=(3, 32, 32))
for r in report.recommendations:
    print(r)
for w in report.warnings:
    print(w)
```

### `edge_of_chaos_sigma_w(activation, at_init=False) -> float`

Compute critical σ_w for an activation function.

```python
sw = edge_of_chaos_sigma_w('gelu')             # Fixed-point: 1.982
sw = edge_of_chaos_sigma_w('gelu', at_init=True)  # Init-time: 1.481
```

### `make_mlp(input_dim, hidden_dim, output_dim, depth, activation) -> nn.Module`

Create an MLP for testing.

### `make_transformer(d_model, n_heads, n_layers, d_ff) -> nn.Module`

Create a Transformer encoder for testing.

---

## PhaseKit-NG API (Non-Gaussian Extension)

The NG extension provides three additional functions that achieve 96% phase
classification accuracy (vs 75% standard) by tracking kurtosis accumulation
through depth.

### `classify_ng(activation, sigma_w, ...) -> NGPhaseResult`

Non-Gaussian phase classification with kurtosis correction.

```python
from phasekit import classify_ng

result = classify_ng("gelu", sigma_w=1.2, width=128, depth=10)
print(result.phase)              # 'ordered', 'critical', or 'chaotic'
print(result.confidence)         # Classification confidence (0-1)
print(result.chi_eff_total)      # Kurtosis-corrected effective chi
print(result.chi_standard)       # Standard chi_1^L for comparison
print(result.kurtosis_accumulated)  # Accumulated excess kurtosis at output
print(result.kurtosis_trajectory)   # Per-layer kurtosis values
print(result.mc_phase)           # MC empirical verification result
```

**Parameters:**
- `activation: str` — Activation function name
- `sigma_w: float = 1.414` — Weight initialization scale
- `sigma_b: float = 0.0` — Bias initialization scale
- `width: int = 128` — Network width
- `depth: int = 10` — Network depth
- `use_mc: bool = True` — Use Monte Carlo empirical verification
- `mc_trials: int = 8` — Number of MC trials
- `seed: int = 42`

**Returns `NGPhaseResult`** with fields: `phase`, `probabilities`, `chi_eff_total`,
`chi_eff_per_layer`, `chi_standard`, `kurtosis_accumulated`, `kurtosis_trajectory`,
`variance_trajectory`, `chi_eff_trajectory`, `mc_phase`, `mc_var_ratio`,
`confidence`, `correction_magnitude`.

### `initialize_ng(dims, activation, seed) -> GSInitResult`

Gradient-searched initialization (GS-CaliKit). Eliminates divergence failures
at depth ≥ 10 by using empirical σ_w search instead of theory-only gain.

```python
from phasekit import initialize_ng

result = initialize_ng([784, 256, 256, 256, 256, 256, 10], "gelu")
print(result.gain_searched)       # Empirically optimal gain
print(result.gain_theory)         # Theory-predicted gain (for comparison)
print(result.var_trajectory)      # Variance through network
print(result.kurtosis_trajectory) # Kurtosis through network
```

**Parameters:**
- `dims: list[int]` — Layer dimensions `[input, hidden1, ..., output]`
- `activation: str = "relu"` — Activation function name
- `seed: int = 42`

**Returns `GSInitResult`** with fields: `method`, `weights`, `biases`,
`sigma_w_per_layer`, `gain_theory`, `gain_searched`, `var_trajectory`,
`kurtosis_trajectory`, `description`.

### `find_critical_sigma_w_ng(activation, width, depth, sigma_b) -> float`

Find critical σ_w using kurtosis-corrected theory. More accurate than standard
`edge_of_chaos_sigma_w` at depth ≥ 10 and for GELU/SiLU.

```python
from phasekit import find_critical_sigma_w_ng

sw = find_critical_sigma_w_ng("gelu", width=128, depth=10)
print(f"Critical σ_w for GELU at depth 10: {sw:.4f}")
```

---

### `DAGAnalysisResult` (dag_propagator.py)
DAG-based variance propagation engine. Used internally by the unified API.

```python
from dag_propagator import analyze_dag
result = analyze_dag(model, input_shape=(3, 32, 32))
```

Returns `DAGAnalysisResult` with:
- `phase`: "ordered", "critical", or "chaotic"
- `chi_total`: Total susceptibility
- `predicted_variance`: Dict of per-node predicted second moments
- `empirical_variance`: Dict of per-node empirical second moments
- `variance_error_pct`: Mean relative error (%)
- `n_nodes`, `n_weight_layers`, `n_branches`, `n_residual`
- `per_layer_recommendations`: Dict of per-layer σ_w for criticality

**Supported layer types:** Linear, Conv1d/2d/3d, ReLU, GELU, SiLU, Tanh, Sigmoid, ELU, LeakyReLU, Mish, LayerNorm, BatchNorm, GroupNorm, MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer, Dropout, AvgPool, MaxPool, Embedding, LSTM, GRU.

**Merge types:** ADD (residual/skip), CAT (concatenation), MUL (gating).

### `MeanFieldAnalyzer` (mean_field_theory.py)
Core mean-field analysis for MLPs.

```python
from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec
mf = MeanFieldAnalyzer()
spec = ArchitectureSpec(depth=20, width=256, activation="relu", sigma_w=1.414)
report = mf.analyze(spec)
```

### `TransformerMeanField` (transformer_mean_field.py)
Mean-field analysis for Transformers.

```python
from transformer_mean_field import TransformerMeanField, TransformerSpec
tmf = TransformerMeanField()
spec = TransformerSpec(n_layers=12, d_model=768, n_heads=12, d_ff=3072)
report = tmf.analyze(spec)
```

### `StochasticCrossoverAnalyzer` (stochastic_crossover.py)
Phase boundary crossover analysis.

```python
from stochastic_crossover import StochasticCrossoverAnalyzer
analyzer = StochasticCrossoverAnalyzer(n_trials=100)
result = analyzer.analyze_crossover("tanh", width=128, depth=10)

scaling = analyzer.analyze_width_scaling("relu", widths=[32, 64, 128, 256, 512])
print(f"Exponent: {scaling['scaling_exponent']:.3f}")  # Should be ~0.5
```

**`analyze_crossover(activation, width, depth) -> CrossoverResult`**
Returns `CrossoverResult` with: `crossover_width`, `chi1_std_empirical`, `chi1_std_analytical`.

**`analyze_width_scaling(activation, widths) -> dict`**
Returns dict with: `scaling_exponent`, `scaling_prefactor`, `r_squared`.

## Data Classes

- `ArchitectureSpec`: MLP/ConvNet/ResNet specification
- `TransformerSpec`: Transformer specification  
- `MFReport`: Analysis results for MLPs
- `TransformerMFReport`: Analysis results for Transformers
- `PhaseClassification`: Phase with posterior probabilities

## Supported Activations

`relu`, `tanh`, `gelu`, `silu`, `leaky_relu`, `elu`, `mish`, `sigmoid`, `linear`

## Experiment Scripts

| Script | Description |
|---|---|
| `run_developer_validation.py` | Developer API validation (49 architectures incl. 15 HuggingFace models) |
| `run_e2e_validation.py` | End-to-end training validation (120 configs, init comparison, UQ, κ₄) |
| `run_ng_benchmarks.py` | PhaseKit-NG benchmark suite (336 configs, phase + init + kurtosis) |
| `run_comprehensive_validation.py` | Full validation: phase, DAG, init |
| `run_stochastic_crossover.py` | Crossover width scaling |

---

## CLI: Checkpoint, Torchvision, ONNX, and HuggingFace Loading

The CLI supports loading existing models for analysis via `--checkpoint`,
`--from-torchvision`, `--onnx`, `--hf-model`, and `--hf-model-local` flags.

### `--checkpoint PATH`

Load a saved `.pt`/`.pth` file containing a full `nn.Module` (not a state_dict).

```bash
# Diagnose a saved model
python -m src.cli diagnose --checkpoint model.pt

# Use with compute
python -m src.cli compute --checkpoint model.pt --profile quick
```

The file must be saved with `torch.save(model, path)`. If a state_dict is detected, a helpful error message is shown.

### `--from-torchvision NAME`

Load a torchvision model by name with random initialization.

```bash
python -m src.cli diagnose --from-torchvision resnet18
python -m src.cli diagnose --from-torchvision vgg11
python -m src.cli compute --from-torchvision mobilenet_v2 --profile quick
```

Requires `torchvision` to be installed.

### `--onnx PATH`

Load an ONNX model file and extract architecture information.

```bash
python -m src.cli diagnose --onnx model.onnx
python -m src.cli compute --onnx model.onnx --profile quick
```

Requires `onnx` to be installed (`pip install onnx`).

### `--hf-model NAME`

Load a HuggingFace model by name and extract architecture information.

```bash
python -m src.cli diagnose --hf-model bert-base-uncased
python -m src.cli diagnose --hf-model gpt2
python -m src.cli compute --hf-model bert-base-uncased --profile quick
```

Requires `transformers` to be installed (`pip install transformers`).

### `--hf-model-local PATH`

Load architecture information from a local HuggingFace checkpoint directory
(reads `config.json` without network access).

```bash
python -m src.cli diagnose --hf-model-local ./my-hf-checkpoint/
```

### `--input-shape SHAPE`

Override auto-detected input shape (comma-separated integers, no batch dim).

```bash
python -m src.cli diagnose --checkpoint model.pt --input-shape 3,32,32
```

If omitted, input shape is inferred from the first Linear/Conv layer.

### `diagnose` subcommand

Runs phase classification and prints a diagnostic report.

```bash
python -m src.cli diagnose --checkpoint model.pt
python -m src.cli diagnose --from-torchvision resnet18 --input-shape 3,224,224
```

### Checkpoint Loader API (`checkpoint_loader.py`)

```python
from src.checkpoint_loader import load_checkpoint, load_torchvision_model, detect_input_shape
from src.checkpoint_loader import load_onnx_model, load_hf_model, load_hf_model_local

model = load_checkpoint("model.pt")
shape = detect_input_shape(model)

model = load_torchvision_model("resnet18")
```

### `load_onnx_model(path) -> dict`

Extract architecture description from an ONNX file. Returns a dict with
`n_layers`, `d_model`, `n_linear`, `n_conv`, `op_counts`, and `initializer_shapes`.

```python
info = load_onnx_model("model.onnx")
print(info["n_layers"], info["d_model"])
```

Requires `pip install onnx`.

### `load_hf_model(name, load_weights=False) -> dict`

Extract architecture description from a HuggingFace model identifier.
Returns a dict with `n_layers`, `d_model`, `d_ff`, `activation`. When
`load_weights=True` the dict also contains a `"model"` key with the
loaded `nn.Module`.

```python
info = load_hf_model("bert-base-uncased")
print(info["n_layers"], info["d_model"], info["activation"])
```

Requires `pip install transformers`.

### `load_hf_model_local(path) -> dict`

Read `config.json` from a local HuggingFace checkpoint directory.
Returns the same dict shape as `load_hf_model` without network access.

```python
info = load_hf_model_local("./my-hf-checkpoint/")
```

### Empirical Kurtosis (`empirical_kurtosis.py`)

Monte Carlo estimation of excess kurtosis for moment-closure validation.

```python
from empirical_kurtosis import compute_empirical_moments, compare_analytical_vs_empirical

# Compare analytical vs empirical kappa4
results = compare_analytical_vs_empirical(
    activations=["relu", "tanh", "gelu", "silu"],
    widths=[128, 512],
    sigma_w_values=[1.0, 2.0],
    n_layers=10,
    n_samples=100
)
```
