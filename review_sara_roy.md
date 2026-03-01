# Review: Finite-Width Phase Diagrams for Neural Network Training Dynamics

**Reviewer:** Sara Roy (Machine Learning and Formal Verification Expert)
**Score: 7/10**

---

## Summary

This work constructs a system that predicts the dynamical training regime of neural networks by computing finite-width phase diagrams, validated against ground-truth training runs. The system bridges theoretical ML (NTK, mean-field, µP) with experimental validation, aiming to provide practitioners with actionable hyperparameter guidance.

## Strengths

The evaluation methodology is exemplary in its rigor. The multi-criteria ground-truth classification — requiring ≥3/4 agreement among kernel alignment drift, representation similarity, weight displacement norm, and linear probing accuracy gap — addresses the fundamental ambiguity in regime classification that plagues the training dynamics literature. This is a significant methodological contribution independent of the theoretical framework, as most prior work uses single indicators (typically kernel alignment alone) which can be misleading near boundaries.

The ablation study design is well-constructed: isolating 1/N corrections (ON vs. OFF), comparing single-backend vs. multi-backend fusion, and testing Nyström rank sensitivity directly measures the marginal contribution of each novel component. The inclusion of five baselines — including the non-theoretical heuristic (predict rich if η·N^{1-a-b} > C) — provides a meaningful lower bound that tests whether the theoretical machinery adds value beyond curve-fitting.

The claim structure is falsifiable and honestly scoped: AUC > 0.90, boundary localization within 15% relative error for >80% of configurations, and >90% of failures flagged by UQ. These are specific enough to be meaningfully tested and modest enough to be credible.

## Weaknesses

The generalization story has a critical gap: the system is validated only on MNIST, Fashion-MNIST, CIFAR-10 (subsampled to 2K with PCA to 100 dimensions), and two tabular datasets. These are toy datasets by modern standards, and the PCA preprocessing of CIFAR-10 fundamentally alters its spectral structure — the kernel-task alignment computed on PCA-reduced data may not reflect the alignment on the original high-dimensional data. The system's predictions on these preprocessed datasets may not transfer to realistic settings.

The training data quality for ground-truth evaluation is concerning. With only 5 seeds for non-boundary points and 20 seeds for boundary-adjacent points, the variance in regime classification may be substantial, particularly for stochastic gradient descent with batch noise. The system's claimed 15% boundary localization accuracy requires that ground-truth boundaries themselves be localized with comparable or better precision, and no analysis demonstrates this.

The ML methodology for regime classification relies entirely on hand-crafted indicators (kernel alignment, weight displacement, etc.) with threshold-based classification. These thresholds are not learned from data — they are either theoretically motivated or empirically chosen. This introduces a circularity risk: if the thresholds are tuned to match the system's predictions, the evaluation becomes self-confirming.

The scope limitation to non-transformer architectures (ConvNets, ResNets, MLPs) substantially reduces the practical value for current ML practice. While the mathematical justification is valid (softmax attention breaks the analysis), practitioners who most need hyperparameter guidance are training transformers, not ResNets.

## Recommendations

To strengthen the ML methodology: (1) validate ground-truth classification robustness by showing that increasing seed count from 5→50 does not substantially change boundary locations, (2) evaluate on at least one dataset without PCA reduction to test spectral fidelity, (3) demonstrate that ground-truth classification thresholds are not overfit to the system's predictions via a hold-out protocol, and (4) discuss a concrete path toward transformer support.
