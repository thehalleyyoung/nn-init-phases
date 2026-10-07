# Reproducing the results

## Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy scipy torch pytest hypothesis matplotlib modal
```

The independent audit uses Python 3.14.7 and torch 2.11.0; historical training
uses the pinned Modal image below. The Modal images
pin their own versions (see `experiments/trainability/modal_app.py`):

* training/predictors: Python 3.11, torch 2.5.1, torchvision 0.20.1, numpy 1.26.4,
  scipy 1.13.1, NVIDIA L4 GPU (training) or CPU containers (predictors);
* AutoInit: Python 3.11, tensorflow-cpu 2.12.1, tensorflow-addons 0.21.0,
  `cognizant-ai-labs/autoinit@03ce3416b02a076696f8cc4302a9910393dc3c5d`.

Third-party code fetched into the training image at pinned commits:
`zhuchen03/gradinit@777c4070c86bdf575c8df6254dbbab983173ef4f` (GradInit, MetaInit),
`ducha-aiki/LSUV-pytorch@60e9fee5f15e0d4edd1f4d9f0587c7149c1974ed` (LSUV).
CIFAR-10 is downloaded by torchvision during the image build
(https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz; torchvision checks MD5
`c58f30108f718f92721af3b95e74349a`).

## Commands

All commands run from the repository root. Each Modal command appends to
`results/trainability/<exp>.jsonl` and skips cells already present, so an interrupted
run can be resumed.

```bash
# E1: predictors on CPU, then training of 162 cells x 3 seeds x lr grid (1,296 runs)
modal run experiments/trainability/modal_app.py --exp e1_predict
modal run experiments/trainability/modal_app.py --exp e1
python experiments/trainability/analyze_e1.py          # -> e1_summary.json, e1_table.csv, paper/figures/e1_grid.png

# E2: AutoInit weights (TensorFlow, CPU), then 72 cells (8 networks x 9 schemes)
modal run experiments/trainability/modal_app.py --exp autoinit
modal run experiments/trainability/modal_app.py --exp e2
python experiments/trainability/analyze_e2.py          # -> e2_summary.json, e2_table.csv

# Paper numbers and PDF
python experiments/trainability/make_paper_numbers.py  # -> paper/numbers.tex, paper/e2_table_body.tex
cd paper && pdflatex tool_paper.tex && pdflatex tool_paper.tex

# Audit of the earlier Monte Carlo benchmark (CPU)
modal run experiments/trainability/modal_app.py --exp audit_ng   # -> results/audit/ng_benchmark_audit.json

# Tests
python -m pytest tests -q
```

The grids, label definition and learning-rate grids are fixed in
`experiments/trainability/configs.py`. One E2 cell (CNN tanh, PhaseKit `fix`) was not run.

A CPU rerun of the MLP-tanh depth-20 `phasekit_initialize` cell contributes nine
rows, covering three seeds and three learning rates. `analyze_e2.py` replaces
that complete Modal cell with these rows. The paper's E2 predictor evaluation
uses the seven external schemes, with three independently initialized seeds
per architecture and scheme.

```bash
OPENBLAS_NUM_THREADS=1 python3 study_audit/reaggregate.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 study_audit/rng_scale_stress.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 study_audit/input_scale_stress.py /path/to/cifar/data
python3 -m pytest tests/test_input_variance.py tests/test_rng_isolation.py
```

The input-scale study performs autograd on 32 real CIFAR-10 images without
training. Its archive and parameter hashes are in the raw JSON receipt.

**E2 cost-log reconciliation.** `results/modal_costs.jsonl` has 73 `e2` entries: one
zero-cost placeholder for the launch that failed before running any cell (module name
clash), and 72 completed `e2_group` calls. Those are the 71 cells kept in `e2.jsonl`
plus one GradInit cell (MLP tanh, L = 20) that was run from PyTorch's default init and
discarded when GradInit was switched to the GradInit repository's Kaiming start; its 9
rows were removed from `e2.jsonl`. Other cells in flight when that launch was stopped
did not return results and are not logged.

**Provenance.** `results/trainability/provenance_sha256.txt` holds SHA-256 hashes of the
CIFAR-10 archive and of the third-party files at their pinned commits, computed locally
from the same URLs the Modal image uses.
`results/trainability/autoinit_weights.jsonl` (1.06 GB) is not committed;
`autoinit_weight_stds.jsonl` records its per-layer weight scales and SHA-256.

## Recorded outputs

| File | Content |
|---|---|
| `results/trainability/e1_predict.jsonl` | every predictor's output per network |
| `results/trainability/e1.jsonl` | per-run loss/accuracy curves (1,296 runs) |
| `results/trainability/e1_summary.json`, `e1_table.csv` | E1 metrics and per-network table |
| `results/trainability/e2.jsonl`, `e2_summary.json`, `e2_table.csv` | E2 runs and summary |
| `results/trainability/smoke*.jsonl` | timing smoke runs (not used in the paper) |
| `results/audit/ng_benchmark_audit.json` | recomputation of the Monte Carlo benchmark |
| `results/tests/` | full pytest output and JUnit XML |
| `results/modal_costs.jsonl` | per-call wall time and estimated cost |
| `results/modal_billing.json` | Modal's billed cost for the project's apps |

Audit E1 confidence intervals group all scales and seeds in18 architecture/activation/depth
clusters; E2 groups the seven external initializers and seeds in8 architecture clusters.
The matched fixed-point comparison uses `L/xi <= ln(1e4)` without label tuning.
New scalar quantities are traceable to `results/audit/independent-reanalysis.json` and
`input-variance-followup.json`; historical paper macros come from `make_paper_numbers.py`.
