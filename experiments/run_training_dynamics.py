"""
Training dynamics validation: verify that PhaseKit's phase predictions correlate with
actual training outcomes across diverse architectures.

For each architecture:
  1. PhaseKit predicts the phase and recommends sigma_w*
  2. We train with: (a) default init, (b) PhaseKit-recommended init, (c) Xavier, (d) Kaiming
  3. Compare early training loss to validate that critical phase init trains faster

This is the key experiment: if phase predictions matter, PhaseKit init should
outperform random init when the default is far from critical.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'training_dynamics')
os.makedirs(RESULTS_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def apply_phasekit_init(model, input_shape):
    """Apply PhaseKit's recommended initialization."""
    import torch
    import torch.nn as nn
    from dag_propagator import analyze_dag, recommend_sigma_w_dag

    result = analyze_dag(model, input_shape, n_samples=32)

    # Apply per-layer sigma_w recommendations
    for name, module in model.named_modules():
        dag_name = f"mod:{name}"
        if dag_name in result.per_layer_recommendations:
            rec_sw = result.per_layer_recommendations[dag_name]
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                std = rec_sw / np.sqrt(fan_in)
                nn.init.normal_(module.weight, 0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                fan_in = int(np.prod(module.weight.shape[1:]))
                std = rec_sw / np.sqrt(fan_in)
                nn.init.normal_(module.weight, 0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    return result


def apply_xavier_init(model):
    """Apply Xavier/Glorot initialization."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def apply_kaiming_init(model):
    """Apply Kaiming/He initialization."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def train_model(model, input_shape, n_steps=200, lr=0.01, seed=42):
    """Train model and return loss trajectory."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n_samples = 128
    x = torch.randn(n_samples, *input_shape)

    model.eval()
    with torch.no_grad():
        try:
            out = model(x[:32])
        except Exception:
            return None
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == 3:
        n_classes = out.shape[-1]
    else:
        n_classes = out.shape[-1]
    y = torch.randint(0, max(n_classes, 2), (n_samples,))

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        try:
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            if pred.dim() == 3:
                pred = pred[:, -1, :]
            loss = criterion(pred, y[:pred.shape[0]])
            if torch.isnan(loss) or torch.isinf(loss):
                losses.append(float('inf'))
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        except Exception:
            losses.append(float('inf'))
            break

    return losses


def build_test_configs():
    """Build diverse architecture configs for training validation."""
    import torch.nn as nn

    configs = []

    # MLPs with different depths and activations
    for depth, act_cls, act_name in [
        (5, nn.ReLU, 'relu'), (10, nn.ReLU, 'relu'), (15, nn.ReLU, 'relu'),
        (5, nn.GELU, 'gelu'), (10, nn.GELU, 'gelu'), (15, nn.GELU, 'gelu'),
        (5, nn.SiLU, 'silu'), (10, nn.SiLU, 'silu'),
        (5, nn.Tanh, 'tanh'), (10, nn.Tanh, 'tanh'),
    ]:
        def make_mlp(d=depth, a=act_cls):
            layers = [nn.Linear(128, 256), a()]
            for _ in range(d - 2):
                layers.extend([nn.Linear(256, 256), a()])
            layers.append(nn.Linear(256, 10))
            return nn.Sequential(*layers)
        configs.append((f"MLP-{depth}L-{act_name}", make_mlp, (128,), "MLP"))

    # CNNs
    class SimpleCNN(nn.Module):
        def __init__(self, depth=5, channels=32):
            super().__init__()
            layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU()]
            for _ in range(depth - 1):
                layers.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU()])
            layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 10)])
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    for d in [3, 5, 8]:
        configs.append((f"CNN-{d}L", lambda d=d: SimpleCNN(d), (3, 32, 32), "CNN"))

    # ResNets
    class ResBlock(nn.Module):
        def __init__(self, dim, act_cls=nn.ReLU):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.act = act_cls()
            self.fc2 = nn.Linear(dim, dim)
        def forward(self, x):
            return x + self.fc2(self.act(self.fc1(x)))

    class SimpleResNet(nn.Module):
        def __init__(self, dim=128, blocks=4, act_cls=nn.ReLU):
            super().__init__()
            self.proj = nn.Linear(128, dim)
            self.blocks = nn.ModuleList([ResBlock(dim, act_cls) for _ in range(blocks)])
            self.out = nn.Linear(dim, 10)
        def forward(self, x):
            x = self.proj(x)
            for b in self.blocks:
                x = b(x)
            return self.out(x)

    for blocks in [4, 8]:
        configs.append((f"ResNet-{blocks}blk", lambda b=blocks: SimpleResNet(blocks=b), (128,), "ResNet"))
    configs.append(("ResNet-GELU-4blk", lambda: SimpleResNet(blocks=4, act_cls=nn.GELU), (128,), "ResNet"))

    # Transformers
    class PreLNBlock(nn.Module):
        def __init__(self, d=128, nhead=4):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        def forward(self, x):
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
            x = x + self.ff(self.ln2(x))
            return x

    class SimpleTF(nn.Module):
        def __init__(self, n_layers=2, d=128, nhead=4):
            super().__init__()
            self.blocks = nn.ModuleList([PreLNBlock(d, nhead) for _ in range(n_layers)])
            self.out = nn.Linear(d, 10)
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return self.out(x[:, 0])

    for layers in [2, 4]:
        configs.append((f"Transformer-{layers}L", lambda l=layers: SimpleTF(l), (16, 128), "Transformer"))

    return configs


def run_experiment():
    """Run training dynamics validation."""
    import torch
    import copy
    from dag_propagator import analyze_dag

    print("=" * 70)
    print("PhaseKit Training Dynamics Validation")
    print("=" * 70)

    configs = build_test_configs()
    print(f"\nTotal configs: {len(configs)}")

    results = []
    pk_wins = 0
    pk_total = 0

    for i, (name, model_fn, input_shape, family) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {name}...")

        result = {"name": name, "family": family}

        try:
            # Create fresh models with different inits
            model_default = model_fn()
            model_phasekit = copy.deepcopy(model_default)
            model_xavier = copy.deepcopy(model_default)
            model_kaiming = copy.deepcopy(model_default)

            # Get PhaseKit analysis
            dag_result = analyze_dag(model_default, input_shape, n_samples=32)
            result["predicted_phase"] = dag_result.phase
            result["predicted_chi"] = dag_result.chi_total
            result["recommended_sigma_w"] = dag_result.recommended_sigma_w

            # Apply inits
            pk_result = apply_phasekit_init(model_phasekit, input_shape)
            apply_xavier_init(model_xavier)
            apply_kaiming_init(model_kaiming)

            # Train each
            init_methods = {
                "default": model_default,
                "phasekit": model_phasekit,
                "xavier": model_xavier,
                "kaiming": model_kaiming,
            }

            for method_name, model in init_methods.items():
                losses = train_model(model, input_shape, n_steps=200, lr=0.01)
                if losses:
                    result[f"{method_name}_final_loss"] = losses[-1]
                    result[f"{method_name}_losses"] = losses[::10]  # subsample
                    result[f"{method_name}_converged"] = losses[-1] < losses[0] * 0.9
                else:
                    result[f"{method_name}_final_loss"] = float('inf')

            # Determine winner
            final_losses = {m: result.get(f"{m}_final_loss", float('inf'))
                          for m in init_methods.keys()}
            valid_losses = {m: l for m, l in final_losses.items() if np.isfinite(l)}
            if valid_losses:
                winner = min(valid_losses, key=valid_losses.get)
                result["winner"] = winner
                if winner == "phasekit":
                    pk_wins += 1
                pk_total += 1

                print(f"  Phase={dag_result.phase}, χ={dag_result.chi_total:.4f}")
                for m in ["default", "phasekit", "xavier", "kaiming"]:
                    fl = result.get(f"{m}_final_loss", float('inf'))
                    tag = " ← BEST" if m == winner else ""
                    print(f"    {m:10s}: {fl:.4f}{tag}")

        except Exception as e:
            result["error"] = str(e)
            print(f"  ERROR: {e}")

        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING DYNAMICS SUMMARY")
    print("=" * 70)
    print(f"PhaseKit wins: {pk_wins}/{pk_total} = {100*pk_wins/max(pk_total,1):.1f}%")

    # Count per-family wins
    family_wins = {}
    for r in results:
        fam = r.get("family", "unknown")
        if fam not in family_wins:
            family_wins[fam] = {"total": 0, "pk_wins": 0, "xavier_wins": 0, "kaiming_wins": 0}
        winner = r.get("winner", "none")
        family_wins[fam]["total"] += 1
        if winner == "phasekit":
            family_wins[fam]["pk_wins"] += 1
        elif winner == "xavier":
            family_wins[fam]["xavier_wins"] += 1
        elif winner == "kaiming":
            family_wins[fam]["kaiming_wins"] += 1

    for fam, stats in sorted(family_wins.items()):
        print(f"  {fam:15s}: PK={stats['pk_wins']}, Xavier={stats['xavier_wins']}, Kaiming={stats['kaiming_wins']} / {stats['total']}")

    # Phase-stratified analysis
    print("\nPhase-stratified analysis:")
    for phase in ["ordered", "critical", "chaotic"]:
        phase_results = [r for r in results if r.get("predicted_phase") == phase and "winner" in r]
        if phase_results:
            n = len(phase_results)
            pk = sum(1 for r in phase_results if r["winner"] == "phasekit")
            print(f"  {phase:10s}: PhaseKit wins {pk}/{n} = {100*pk/n:.0f}%")

    # Save
    output = {
        "experiment": "training_dynamics_validation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_configs": len(configs),
        "pk_wins": pk_wins,
        "pk_total": pk_total,
        "pk_win_rate": pk_wins / max(pk_total, 1),
        "family_wins": family_wins,
        "results": results,
    }

    out_path = os.path.join(RESULTS_DIR, "training_dynamics_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_experiment()
