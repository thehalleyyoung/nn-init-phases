# Review: Finite-Width Phase Diagrams for Neural Network Training Dynamics

**Reviewer:** Yuan Cheng (Probabilistic Modeling Researcher)
**Score: 7/10**

---

## Summary

This work proposes the first integrated system for computing finite-width phase diagrams predicting the lazy-to-rich transition boundary in neural network training dynamics. The system unifies NTK theory, mean-field limits, and maximal update parameterization (µP) into a compiler pipeline that takes architecture specifications and produces regime classifications validated against ground-truth training runs. The core mathematical contribution is the derivation of 1/N corrections to the NTK for convolutional and residual architectures, extending Dyer & Gur-Ari's perturbative framework.

## Strengths

The treatment of uncertainty quantification is unusually honest for a theoretical ML system. The authors explicitly track three sources of approximation error — perturbative validity (|Θ⁽¹⁾/Θ⁽⁰⁾| monitoring), moment-closure quality (κ₄ magnitude), and Nyström approximation error — and flag predictions that exceed calibrated thresholds as low-confidence. The target of identifying >90% of prediction failures as low-confidence demonstrates genuine statistical self-awareness. The formulation of the lazy-to-rich transition via the dimensionless coupling parameter γ = η · N^{-(1−a−b)} is elegant and connects parameterization choices directly to convergence properties of the perturbative expansion.

The evaluation plan's use of multi-criteria ground-truth classification (kernel alignment drift, representation similarity, weight displacement, linear probing accuracy gap with a ≥3/4 conjunction rule) is methodologically sound and avoids the single-metric pitfalls that plague regime classification studies.

## Weaknesses

The Gaussian moment-closure assumption (κ₄ = 0) is the most significant probabilistic concern. While the authors acknowledge this is "an approximation ansatz, not a theorem," the ±50% sensitivity analysis on κ₄ is insufficient to characterize the bias. In regions near phase boundaries — precisely where the system's predictions matter most — the distribution of pre-activations is known to be non-Gaussian (Hanin & Nica, 2020), and κ₄ perturbations of ±50% may not span the true range. A more rigorous approach would derive data-dependent bounds on κ₄ from the architecture's combinatorial structure.

The convergence properties of the 1/N expansion itself deserve more scrutiny. The one-loop vs. two-loop convergence check is a necessary but not sufficient condition for expansion validity. The perturbation series may be asymptotic rather than convergent, and the system provides no estimate of the radius of convergence in N. For widths where the expansion formally converges but the asymptotic series has begun to diverge, the UQ module may produce overconfident predictions.

The Nyström approximation introduces O(1/m) error that propagates through the entire pipeline, yet the interaction between Nyström rank selection and phase boundary accuracy is tested only via ablation over four rank values {100, 500, 1000, 2000}. For datasets with heavy-tailed spectral distributions, the rank required for accurate kernel-task alignment may be substantially higher, and the error propagation through the bifurcation detector is nonlinear and potentially amplifying.

The stochastic aspects of the evaluation — 20-seed training for boundary-adjacent points, 5 seeds elsewhere — may be underpowered for detecting phase boundaries with stochastic gradient noise. The variance of regime classification across seeds near the boundary could mask the transition, and no power analysis is provided to justify these sample sizes.

## Recommendations

The project would benefit from (1) deriving architecture-specific bounds on κ₄ rather than relying on perturbation sensitivity analysis, (2) providing convergence radius estimates for the 1/N expansion as a function of architecture parameters, and (3) conducting a power analysis for the seed counts used in ground-truth evaluation. The honest treatment of limitations is commendable, but several of the distributional assumptions merit deeper probabilistic analysis.
