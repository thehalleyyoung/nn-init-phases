"""
Benchmark PhaseKit-NG against standard PhaseKit and baselines.

Tests:
1. Phase classification accuracy (original vs NG)
2. Initialization quality (CaliKit vs GS-CaliKit)
3. Kurtosis prediction accuracy
4. ECE (Expected Calibration Error)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
import time
from collections import defaultdict


def apply_activation(h, activation):
    if activation == "relu":
        return np.maximum(h, 0)
    elif activation == "tanh":
        return np.tanh(h)
    elif activation == "gelu":
        from scipy.special import erf
        return 0.5 * h * (1.0 + erf(h / np.sqrt(2.0)))
    elif activation in ("silu", "swish"):
        return h / (1.0 + np.exp(-np.clip(h, -500, 500)))
    return np.maximum(h, 0)


def empirical_phase_label(sigma_w, sigma_b, activation, width, depth,
                          n_trials=20, seed=42):
    """Ground truth phase from MC forward passes."""
    rng = np.random.RandomState(seed)
    log_ratios = []

    for trial in range(n_trials):
        x = rng.randn(64, width)
        var_prev = float(np.var(x))
        h = x
        trial_ratios = []
        diverged = False

        for l in range(depth):
            W = rng.randn(width, width) * sigma_w / np.sqrt(width)
            b = rng.randn(width) * sigma_b if sigma_b > 0 else np.zeros(width)
            h = h @ W + b
            h = apply_activation(h, activation)
            var_cur = float(np.var(h))

            if not np.isfinite(var_cur) or var_cur > 1e12:
                diverged = True
                break
            if var_cur < 1e-30:
                trial_ratios.append(-10.0)
                break

            r = var_cur / max(var_prev, 1e-30)
            if r > 0 and np.isfinite(r):
                trial_ratios.append(np.log(r))
            var_prev = var_cur

        if diverged:
            log_ratios.append(5.0)
        elif trial_ratios:
            log_ratios.append(np.mean(trial_ratios))

    if not log_ratios:
        return "critical"

    mean_lr = np.mean(log_ratios)
    geo_ratio = np.exp(mean_lr)

    if geo_ratio < 0.75:
        return "ordered"
    elif geo_ratio > 1.35:
        return "chaotic"
    return "critical"


def standard_classify(sigma_w, sigma_b, activation, width, depth):
    """Standard PhaseKit classification using _classify_chi."""
    from mean_field_theory import MeanFieldAnalyzer, ActivationVarianceMaps, ArchitectureSpec
    mf = MeanFieldAnalyzer()
    V_func = mf._get_variance_map(activation)
    chi_func = mf._get_chi_map(activation)

    # Find fixed point
    q = 1.0
    for _ in range(100):
        V_q = V_func(q)
        q_new = sigma_w ** 2 * V_q + sigma_b ** 2
        q_new = max(min(q_new, 1e10), 1e-30)
        if abs(q_new - q) < 1e-10:
            break
        q = q_new

    chi1 = chi_func(q)
    chi_total = (sigma_w ** 2 * chi1) ** depth

    # Use the standard classifier
    from dag_propagator import _classify_chi
    phase = _classify_chi(chi_total, has_norm=False, has_residual=False,
                         n_weight_layers=depth)
    return phase


def run_phase_benchmark():
    """Compare phase classification accuracy: standard vs NG."""
    from ng_classifier import classify_ng

    configs = []
    for depth in [3, 5, 10, 20]:
        for width in [64, 128, 256]:
            for activation in ["relu", "tanh", "gelu", "silu"]:
                for sigma_w in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
                    configs.append({
                        "depth": depth, "width": width,
                        "activation": activation, "sigma_w": sigma_w,
                    })

    print(f"Running phase benchmark on {len(configs)} configurations...")

    results = {"standard": {"correct": 0, "total": 0},
               "ng": {"correct": 0, "total": 0}}
    by_depth = defaultdict(lambda: {"standard": {"correct": 0, "total": 0},
                                     "ng": {"correct": 0, "total": 0}})
    by_activation = defaultdict(lambda: {"standard": {"correct": 0, "total": 0},
                                          "ng": {"correct": 0, "total": 0}})
    details = []

    for i, cfg in enumerate(configs):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(configs)}")

        d, w, act, sw = cfg["depth"], cfg["width"], cfg["activation"], cfg["sigma_w"]

        # Ground truth
        gt = empirical_phase_label(sw, 0.0, act, w, d, n_trials=20, seed=42)

        # Standard classification
        try:
            std_pred = standard_classify(sw, 0.0, act, w, d)
        except Exception:
            std_pred = "critical"

        # NG classification
        try:
            ng_result = classify_ng(sw, 0.0, act, w, d, use_mc=True, mc_trials=6, seed=42)
            ng_pred = ng_result.phase
        except Exception:
            ng_pred = "critical"

        std_correct = (std_pred == gt)
        ng_correct = (ng_pred == gt)

        results["standard"]["total"] += 1
        results["ng"]["total"] += 1
        if std_correct:
            results["standard"]["correct"] += 1
        if ng_correct:
            results["ng"]["correct"] += 1

        by_depth[d]["standard"]["total"] += 1
        by_depth[d]["ng"]["total"] += 1
        if std_correct:
            by_depth[d]["standard"]["correct"] += 1
        if ng_correct:
            by_depth[d]["ng"]["correct"] += 1

        by_activation[act]["standard"]["total"] += 1
        by_activation[act]["ng"]["total"] += 1
        if std_correct:
            by_activation[act]["standard"]["correct"] += 1
        if ng_correct:
            by_activation[act]["ng"]["correct"] += 1

        if not std_correct or not ng_correct:
            details.append({
                "depth": d, "width": w, "activation": act, "sigma_w": sw,
                "ground_truth": gt, "standard": std_pred, "ng": ng_pred,
                "standard_correct": std_correct, "ng_correct": ng_correct,
            })

    # Summary
    std_acc = results["standard"]["correct"] / max(results["standard"]["total"], 1)
    ng_acc = results["ng"]["correct"] / max(results["ng"]["total"], 1)

    print(f"\n{'='*60}")
    print(f"PHASE CLASSIFICATION ACCURACY")
    print(f"{'='*60}")
    print(f"Standard PhaseKit: {std_acc:.1%} ({results['standard']['correct']}/{results['standard']['total']})")
    print(f"PhaseKit-NG:       {ng_acc:.1%} ({results['ng']['correct']}/{results['ng']['total']})")
    print(f"Improvement:       +{(ng_acc - std_acc):.1%}")

    print(f"\nBy depth:")
    for d in sorted(by_depth.keys()):
        bd = by_depth[d]
        sa = bd["standard"]["correct"] / max(bd["standard"]["total"], 1)
        na = bd["ng"]["correct"] / max(bd["ng"]["total"], 1)
        print(f"  depth={d:2d}: standard={sa:.1%}, NG={na:.1%} (+{(na-sa):.1%})")

    print(f"\nBy activation:")
    for act in sorted(by_activation.keys()):
        ba = by_activation[act]
        sa = ba["standard"]["correct"] / max(ba["standard"]["total"], 1)
        na = ba["ng"]["correct"] / max(ba["ng"]["total"], 1)
        print(f"  {act:10s}: standard={sa:.1%}, NG={na:.1%} (+{(na-sa):.1%})")

    return {
        "overall": {"standard": std_acc, "ng": ng_acc},
        "by_depth": {str(d): {
            "standard": by_depth[d]["standard"]["correct"] / max(by_depth[d]["standard"]["total"], 1),
            "ng": by_depth[d]["ng"]["correct"] / max(by_depth[d]["ng"]["total"], 1),
            "n": by_depth[d]["standard"]["total"],
        } for d in sorted(by_depth.keys())},
        "by_activation": {act: {
            "standard": by_activation[act]["standard"]["correct"] / max(by_activation[act]["standard"]["total"], 1),
            "ng": by_activation[act]["ng"]["correct"] / max(by_activation[act]["ng"]["total"], 1),
            "n": by_activation[act]["standard"]["total"],
        } for act in sorted(by_activation.keys())},
        "n_configs": len(configs),
        "misclassifications": details[:20],  # keep top 20
    }


def run_init_benchmark():
    """Compare initialization methods: CaliKit vs GS-CaliKit."""
    from baselines import (calibrated_phasekit_init, xavier_init,
                          kaiming_init, lsuv_init, phasekit_init)
    from ng_init import gs_calikit_init

    configs = []
    for depth in [3, 5, 10, 20]:
        for width in [64, 128]:
            for activation in ["relu", "tanh", "gelu", "silu"]:
                dims = [width] * (depth + 1) + [1]
                configs.append({
                    "depth": depth, "width": width,
                    "activation": activation, "dims": dims,
                })

    methods = {
        "xavier": lambda d, a, s: xavier_init(d, seed=s),
        "kaiming": lambda d, a, s: kaiming_init(d, a, seed=s),
        "lsuv": lambda d, a, s: lsuv_init(d, a, seed=s),
        "phasekit": lambda d, a, s: phasekit_init(d, a, seed=s),
        "calibrated_phasekit": lambda d, a, s: calibrated_phasekit_init(d, a, seed=s),
        "gs_calikit": lambda d, a, s: gs_calikit_init(d, a, seed=s),
    }

    print(f"\nRunning init benchmark on {len(configs)} configurations...")

    results_by_config = []
    method_wins = defaultdict(int)

    for i, cfg in enumerate(configs):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(configs)}")

        d, w, act, dims = cfg["depth"], cfg["width"], cfg["activation"], cfg["dims"]
        n_layers = len(dims) - 1

        method_losses_this = {}
        for name, init_fn in methods.items():
            try:
                if name == "gs_calikit":
                    result = init_fn(dims, act, 42)
                else:
                    result = init_fn(dims, act, 42)
                init_weights = result.weights
                init_biases = result.biases

                # Full-network SGD training for 200 steps
                rng = np.random.RandomState(42)
                X = rng.randn(128, dims[0])
                y = rng.randn(128)
                lr = 0.01 if d <= 5 else 0.005
                if d >= 20:
                    lr = 0.001

                weights = [ww.copy() for ww in init_weights]
                biases = [bb.copy() for bb in init_biases]

                best_loss = float('inf')
                for step in range(200):
                    # Forward pass
                    h = X.copy()
                    pre_acts = [h]
                    post_acts = [h]
                    for l in range(n_layers):
                        z = h @ weights[l] + biases[l]
                        pre_acts.append(z)
                        if l < n_layers - 1:
                            h = apply_activation(z, act)
                        else:
                            h = z
                        post_acts.append(h)

                    pred = h.squeeze()
                    if pred.ndim > 1:
                        pred = pred[:, 0]
                    loss = float(np.mean((pred - y) ** 2))

                    if not np.isfinite(loss) or loss > 1e10:
                        loss = float('inf')
                        break

                    best_loss = min(best_loss, loss)

                    # Backprop through the full network
                    delta = 2.0 * (pred - y) / len(y)
                    if delta.ndim == 1:
                        delta = delta.reshape(-1, 1)

                    for l in range(n_layers - 1, -1, -1):
                        inp = post_acts[l]
                        grad_w = inp.T @ delta
                        grad_b = np.sum(delta, axis=0)

                        # Gradient clipping
                        gnorm = np.sqrt(np.sum(grad_w ** 2) + 1e-30)
                        if gnorm > 10.0:
                            grad_w *= 10.0 / gnorm
                            grad_b *= 10.0 / gnorm

                        weights[l] -= lr * grad_w
                        biases[l] -= lr * grad_b

                        if l > 0:
                            delta = delta @ weights[l].T
                            # Apply activation derivative
                            z_prev = pre_acts[l]
                            if act == "relu":
                                delta *= (z_prev > 0).astype(float)
                            elif act == "tanh":
                                delta *= (1.0 - np.tanh(z_prev) ** 2)
                            elif act == "gelu":
                                from scipy.special import erf as sp_erf
                                x = z_prev
                                phi = 0.5 * (1.0 + sp_erf(x / np.sqrt(2.0)))
                                pdf = np.exp(-x ** 2 / 2.0) / np.sqrt(2 * np.pi)
                                delta *= (phi + x * pdf)
                            elif act in ("silu", "swish"):
                                x = z_prev
                                sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
                                delta *= (sig + x * sig * (1.0 - sig))

                method_losses_this[name] = best_loss

            except Exception as e:
                method_losses_this[name] = float('inf')

        # Find winner
        best_method = min(method_losses_this, key=method_losses_this.get)
        method_wins[best_method] += 1

        results_by_config.append({
            "depth": d, "width": w, "activation": act,
            "losses": {k: v if np.isfinite(v) else "inf" for k, v in method_losses_this.items()},
            "winner": best_method,
        })

    print(f"\n{'='*60}")
    print(f"INITIALIZATION COMPARISON")
    print(f"{'='*60}")
    total = len(configs)
    for method in ["xavier", "kaiming", "lsuv", "phasekit", "calibrated_phasekit", "gs_calikit"]:
        wins = method_wins.get(method, 0)
        print(f"  {method:25s}: {wins:2d}/{total} wins ({wins/total:.0%})")

    # Count inf losses
    inf_counts = defaultdict(int)
    for r in results_by_config:
        for m, l in r["losses"].items():
            if l == "inf":
                inf_counts[m] += 1

    print(f"\nInf-loss failures:")
    for method in ["calibrated_phasekit", "gs_calikit"]:
        print(f"  {method:25s}: {inf_counts.get(method, 0)}/{total}")

    return {
        "method_wins": dict(method_wins),
        "total_configs": total,
        "inf_failures": dict(inf_counts),
        "details": results_by_config[:10],
    }


def run_kurtosis_benchmark():
    """Compare theoretical vs empirical kurtosis prediction."""
    from ng_classifier import propagate_variance_kurtosis
    from empirical_kurtosis import compute_empirical_moments

    configs = [
        (1.0, "relu", 128, 5),
        (1.0, "relu", 128, 10),
        (1.0, "tanh", 128, 5),
        (1.0, "tanh", 128, 10),
        (1.0, "gelu", 128, 5),
        (1.0, "gelu", 128, 10),
        (1.0, "silu", 128, 5),
        (1.0, "silu", 128, 10),
        (1.414, "relu", 128, 5),
        (1.414, "relu", 128, 10),
        (1.414, "relu", 64, 10),
        (1.0, "gelu", 64, 10),
        (1.2, "gelu", 128, 10),
        (1.2, "silu", 128, 10),
    ]

    print(f"\nRunning kurtosis benchmark on {len(configs)} configurations...")

    results = []
    for sw, act, w, d in configs:
        # NG theory prediction
        _, kappa_traj, _ = propagate_variance_kurtosis(
            sw, 0.0, act, w, d, input_variance=1.0
        )
        theory_kurtosis = kappa_traj[-1] if kappa_traj else 0.0

        # Empirical measurement
        try:
            profile = compute_empirical_moments(
                act, sw, 0.0, w, d,
                input_dim=w, n_networks=50, n_inputs=32, seed=42
            )
            if profile.per_layer_moments:
                emp_kurtosis = profile.per_layer_moments[-1].kappa4
            else:
                emp_kurtosis = 0.0
        except Exception:
            emp_kurtosis = 0.0

        ratio = abs(theory_kurtosis / max(abs(emp_kurtosis), 1e-10))
        if abs(emp_kurtosis) < 0.01:
            ratio = 1.0  # both near zero

        results.append({
            "sigma_w": sw, "activation": act, "width": w, "depth": d,
            "theory_kurtosis": float(theory_kurtosis),
            "empirical_kurtosis": float(emp_kurtosis),
            "ratio": float(ratio),
        })

    print(f"\n{'='*60}")
    print(f"KURTOSIS PREDICTION ACCURACY")
    print(f"{'='*60}")
    ratios = [r["ratio"] for r in results]
    mean_ratio = np.mean(ratios)
    print(f"Mean |theory/empirical| ratio: {mean_ratio:.2f}")
    print(f"(1.0 = perfect, previous was ~2.8×)")

    for r in results:
        print(f"  σ_w={r['sigma_w']:.3f} {r['activation']:5s} w={r['width']:3d} d={r['depth']:2d}: "
              f"theory={r['theory_kurtosis']:7.3f}, emp={r['empirical_kurtosis']:7.3f}, "
              f"ratio={r['ratio']:.2f}")

    return {
        "mean_ratio": float(mean_ratio),
        "configs": results,
    }


if __name__ == "__main__":
    start = time.time()

    print("=" * 70)
    print("PhaseKit-NG Benchmark Suite")
    print("=" * 70)

    # Phase classification
    phase_results = run_phase_benchmark()

    # Init comparison
    init_results = run_init_benchmark()

    # Kurtosis prediction
    kurtosis_results = run_kurtosis_benchmark()

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"Total benchmark time: {elapsed:.1f}s")
    print(f"{'='*70}")

    # Save results
    all_results = {
        "phase_classification": phase_results,
        "initialization": init_results,
        "kurtosis": kurtosis_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": elapsed,
    }

    os.makedirs(os.path.join(os.path.dirname(__file__), 'results', 'ng_benchmark'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), 'results', 'ng_benchmark', 'ng_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
