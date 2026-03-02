# Review: PhaseKit — Finite-Width Phase Diagrams for Neural Network Training Dynamics

**Reviewer:** Vasco Vasconcelos  
**Persona:** Probabilistic Models & Formal Methods Expert  
**Date:** 2026-03-02  

---

## Summary
PhaseKit constructs finite-width phase diagrams for neural network initialization by computing O(1/N) corrections to the infinite-width mean-field theory. The system uses Gaussian moment closure at the 4th cumulant (κ₄ = 0), Nyström-approximated kernel computations, and DAG-based variance propagation to classify initializations into ordered, critical, or chaotic phases. The stochastic crossover analysis demonstrates Δ(N) = A·N^{−0.5} scaling across all tested activations. The system reports 85.7% phase classification accuracy on 210 configurations and 20 Z3-verified algebraic properties.

## Strengths
1. **Principled stochastic crossover analysis.** The demonstration that the phase boundary width scales as Δ(N) ∝ N^{−1/2} across all 9 activation functions is a clean result with solid theoretical backing from central-limit-theorem arguments applied to products of random matrices. The crossover results (scaling exponent 0.500 with negligible error across ReLU, tanh, GELU, SiLU) provide strong evidence that the system correctly captures the leading-order finite-width fluctuation structure.

2. **Well-structured uncertainty quantification framework.** The three-pronged UQ approach — perturbative validity monitoring (|Θ⁽¹⁾/Θ⁽⁰⁾| threshold), moment-closure quality tracking (κ₄ magnitude), and Nyström error propagation — addresses the right error sources. The confidence-stratified output (high/moderate/near-boundary) is the correct design pattern for a system with known approximation limitations. The self-calibration target (>90% of failures flagged) is a testable specification.

3. **Sound variance propagation formalism.** The DAG-based variance composition theorem (additive for residuals, weighted mean for concatenation, multiplicative for gating) is theoretically well-motivated and handles the key architectural patterns correctly. The dominant-path χ computation for multi-branch networks is a reasonable approximation for phase classification, where the most unstable path determines the overall dynamics.

4. **Explicit modeling of the ordered-critical-chaotic trichotomy.** The inclusion of second-order susceptibility χ₂ for bifurcation classification and Lyapunov exponent λ = log(χ₁) for continuous chaos measurement goes beyond simple threshold-based classification. The PhaseClassification dataclass with posterior probabilities over phases is a more principled output than a hard label.

## Weaknesses
1. **Gaussian moment closure introduces systematic, unquantified bias.** Setting κ₄ = 0 is the most restrictive approximation in the pipeline, and its impact is not rigorously bounded. The sensitivity analysis (±50% κ₄ perturbation) measures *local sensitivity* but not *bias*: even if the phase boundary shifts modestly under ±50% perturbation, the actual κ₄ for deep non-ReLU networks may be far from zero. For tanh networks near criticality, κ₄ can grow with depth due to the bounded activation's non-Gaussian output distribution. The paper provides no depth-dependent analysis of κ₄ accumulation, which is precisely where the closure is most likely to fail — deep networks near phase boundaries, the regime of greatest interest.

2. **Nyström approximation conflates two distinct error sources.** The Nyström method introduces both a *sampling error* (from selecting m landmarks out of n data points) and an *approximation error* (from the low-rank kernel approximation). These have different scaling: sampling error is O(1/√m), while approximation error depends on the kernel's eigenvalue decay. For kernels with slow spectral decay (common for complex data), the Nyström error can dominate the finite-width correction it is supposed to measure. The paper does not report the ratio of Nyström approximation error to the O(1/N) correction magnitude — if the former exceeds the latter, the finite-width corrections are noise, not signal.

3. **Uncertainty flags are not demonstrated to be calibrated.** The UQ module is described architecturally but its calibration is not empirically validated in the reported results. The >90% failure-flagging target is stated as a goal, not a demonstrated achievement. Key questions are unanswered: What is the actual false negative rate (failures not flagged)? How does calibration vary by architecture family? Does the calibration hold near phase boundaries where UQ is most needed? Without calibration curves (predicted confidence vs. actual accuracy), the UQ module is an untested heuristic.

4. **The stochastic crossover results are suspiciously clean.** The scaling exponents are 0.500 with error on the order of 10⁻¹⁶ across all activations. This precision suggests the crossover analysis may be computing an analytically derived quantity (the CLT scaling) rather than empirically measuring crossover width from Monte Carlo samples. If so, the "stochastic crossover analysis" validates the analytical formula, not the system's ability to detect crossover in practice. The crossover_results.json confirms this: the boundary standard deviations follow exact 1/√N scaling, consistent with analytical computation rather than empirical sampling.

5. **No posterior predictive checks.** A probabilistic system should be evaluated by its predictive distribution, not just point estimates. The PhaseClassification provides posterior probabilities over phases, but there is no demonstration that these probabilities are well-calibrated — e.g., when the system assigns P(chaotic) = 0.3, do 30% of such configurations actually exhibit chaotic behavior? Without posterior predictive checks, reliability diagrams, or proper scoring rules (Brier score, log loss), the probabilistic output is decorative rather than informative.

6. **Fourth-cumulant truncation introduces correlations that propagate through depth.** The cumulant expansion at each layer generates κ₄ contributions that feed into the next layer's input distribution. Setting κ₄ = 0 at each layer independently ignores this inter-layer correlation. In a 20-layer network, the accumulated κ₄ from early layers modifies the effective input distribution at later layers, and this effect is not captured by the layer-wise Gaussian closure. The result is a systematic underestimate of non-Gaussianity that grows with depth — precisely the setting where phase classification matters most.

## Minor Issues
- The ConfidenceInterval dataclass uses a fixed 95% confidence level without justification — Bayesian credible intervals or bootstrap intervals may be more appropriate for different use cases.
- The MFReport stores variance, χ, and correlation trajectories as lists, but the statistical uncertainty of each trajectory point is not tracked.
- The crossover prefactor A varies by 50× across activations (1.41 for ReLU vs. 0.027 for tanh) — this large variation deserves theoretical explanation.
- The Monte Carlo calibration of the soft phase classifier is mentioned but the number of MC samples and convergence diagnostics are not specified.
- The O(1/N²) corrections using sixth moment terms are mentioned in mean_field_theory.py but the sixth moment computation is not included in the Z3 verification scope.

## Questions for Authors
1. For a 20-layer tanh network at σ_w = σ_w*, what is the magnitude of κ₄ at the final layer as measured by empirical Monte Carlo sampling, compared to the system's assumption of κ₄ = 0? If the discrepancy exceeds 100%, the Gaussian closure is unreliable precisely at criticality.
2. What is the ratio of Nyström approximation error to the O(1/N) finite-width correction for the CIFAR-10 kernel with m = 500 landmarks? If the Nyström error dominates, the corrections are measuring noise.
3. Can you provide calibration plots (predicted probability vs. observed frequency) for the PhaseClassification posterior probabilities? Without these, the probabilistic output cannot be trusted for decision-making.
4. The crossover exponents appear to be analytical. Can you confirm whether the stochastic crossover module performs actual Monte Carlo sampling over random initializations, or whether it computes the CLT-derived analytical formula?
5. Have you considered cumulant-generating-function approaches (e.g., Edgeworth expansion) as an alternative to moment closure? These provide systematically improvable approximations with explicit error terms, which would address the unquantified bias concern.

## Overall Assessment
PhaseKit presents a probabilistically motivated framework with sound theoretical foundations in mean-field theory and well-designed uncertainty quantification architecture. The stochastic crossover analysis and DAG variance propagation are technically solid. However, the central approximation — Gaussian moment closure at κ₄ = 0 — is not rigorously justified for the deep-network, near-critical regime where the tool is most needed. The absence of calibration validation for both the UQ flags and the probabilistic phase classification significantly weakens the practical reliability claims. The suspiciously exact crossover exponents suggest analytical computation masquerading as empirical validation. The work would be substantially strengthened by (a) empirical κ₄ measurements validating the closure assumption, (b) calibration curves for the probabilistic outputs, and (c) clarity about which results are analytical vs. empirical.

**Recommendation:** Weak Accept — The theoretical framework and system architecture are strong, but the probabilistic methodology has significant gaps in validation. The moment-closure concern is addressable, and the UQ calibration is testable — revision could resolve both.

**Confidence:** 4/5
