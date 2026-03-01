# PhaseKit API Reference

## Unified API (`phasekit.py`)

The primary user-facing interface. Three functions cover all use cases.

```python
from phasekit import classify, initialize, diagnose
```

### `classify(model, input_shape, ...) -> ClassifyResult`

Classify the initialization phase of any PyTorch model.

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

## Internal Modules

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
| `run_comprehensive_validation.py` | Full validation: phase, DAG, init |
| `run_universal_graph_experiment.py` | 61-architecture DAG validation |
| `run_z3_finite_width.py` | Z3 finite-width verification |
| `run_transformer_experiments.py` | Transformer validation |
| `run_stochastic_crossover.py` | Crossover width scaling |
| `run_z3_convergence.py` | Z3 SMT convergence verification |
