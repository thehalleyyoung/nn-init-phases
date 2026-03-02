# Review: PhaseKit — Finite-Width Phase Diagrams for Neural Network Training Dynamics

**Reviewer:** Cesar Sanchez  
**Persona:** Formal Verification & AI Researcher  
**Date:** 2026-03-02  

---

## Summary
PhaseKit proposes a computational system that predicts neural network training dynamics at initialization time by computing finite-width phase diagrams, classifying networks into ordered, critical, or chaotic regimes using O(1/N) corrections via the Hessian-Jacobian contraction tensor. The system claims 20 Z3-verified algebraic properties, 85.7% phase classification accuracy across 210 configurations, and DAG-based variance propagation for arbitrary architectures. It unifies NTK theory, mean-field limits, and μP parameterization into a single framework operating without any training.

## Strengths
1. **Novel integration of formal verification into ML theory tooling.** The use of Z3 SMT solving to verify algebraic properties of the finite-width correction hierarchy is commendable and unusual in ML systems work. Properties P13–P20 covering correction sign, magnitude bounds, perturbative validity, and monotonicity provide machine-checked guarantees that manual derivation alone cannot match. The approach of negating the target property and checking for unsatisfiability is correctly applied.

2. **Mathematically principled architecture.** The perturbative expansion Θ(t) = Θ⁽⁰⁾ + (1/N)Θ⁽¹⁾(t) + O(1/N²) is well-grounded in the Dyer & Gur-Ari framework, and extending it to convolutional and residual architectures via per-layer H_{ijk} decomposition is a genuine mathematical contribution. The formulation of phase boundaries as eigenvalue-crossing bifurcation problems is mathematically natural.

3. **Honest limitation disclosure.** The problem statement explicitly acknowledges moment-closure truncation, Nyström approximation errors, exclusion of softmax attention, and empirical depth corrections. This transparency about scope is rare and scientifically appropriate.

4. **100% chaotic precision.** Perfect precision on the chaotic class is practically important because chaotic initialization is the most destructive failure mode. This suggests the classifier is conservative in the right direction.

## Weaknesses
1. **Z3 verification scope is far narrower than suggested.** The 20 verified properties are algebraic identities over symbolic expressions (e.g., P13 verifies that σ_w⁴ · κ₄ · V² / N ≥ 0 when all inputs are positive — a trivially true statement about products of non-negative reals). These are not verification of the *system's correctness* but rather sanity checks on isolated formulas. The critical path — the ODE integration, the DAG traversal, the phase classification logic, the Nyström approximation error propagation — is entirely unverified. Claiming "20 Z3-verified properties" without clarifying this gap risks misleading readers into believing the system has formal guarantees it does not possess.

2. **No soundness proof for the overall pipeline.** The system chains multiple approximations: Gaussian moment closure (κ₄ = 0), Nyström subsampling, one-loop truncation, and empirical depth corrections. Each introduces error, and these errors compose. There is no formal argument — let alone a proof — that the composed error remains bounded. The uncertainty quantification module tracks individual error sources but provides no guarantee about their interaction. A formally verified pipeline would require showing that the UQ flags are *sound* (every true failure is flagged), which is an empirical claim, not a formal one.

3. **The Z3 properties verify necessary but not sufficient conditions.** Property P15 (perturbative validity) proves that if N exceeds a threshold, the correction ratio is below 0.5. But this says nothing about whether the correction *accurately models the true finite-width behavior* — it only says the perturbation series hasn't obviously diverged. The gap between "the series hasn't blown up" and "the prediction is correct" is where the real verification challenge lies, and it is entirely unaddressed.

4. **Property P20 reveals an internal inconsistency.** The code comments for P20 (chi_1 correction vanishes for ReLU) contain an extended internal debate where the implementer discovers that E[φ'⁴] − (E[φ'²])² = 1/4 ≠ 0 for ReLU, contradicting the claimed property. This is verified anyway by restricting to E[φ'⁴] = E[φ'²] (which is true) but conflating it with the variance being zero (which is false). This suggests the Z3 verification encodes the *intended* algebraic statement without confirming it matches the *actual mathematical claim*. If one of 20 properties has a visible inconsistency in its own documentation, what confidence can we have in the other 19?

5. **Convergence claims lack formal treatment.** The system checks one-loop vs. two-loop agreement at target widths, but "agreement" is a numerical threshold, not a formal convergence proof. For the perturbative expansion to be trustworthy, one would need either (a) a rigorous convergence radius estimate for the 1/N series, or (b) a Borel summability argument. Neither is provided. The current approach is closer to numerical spot-checking than to verification.

## Minor Issues
- The moment-closure sensitivity analysis (±50% κ₄ perturbation) is ad hoc — the choice of 50% has no theoretical justification.
- The 85.7% classification accuracy means roughly 1 in 7 predictions is wrong, and the paper does not analyze whether the UQ flags successfully capture these failures.
- Z3 verification times are not reported — if any property required extended solver time, the result may depend on solver heuristics.
- The claim "20/20 (unsat)" suggests all properties are universally valid, but several (P17, P18) are only valid under specific moment formulas that themselves are unverified.
- No mention of whether the Z3 encoding uses exact arithmetic (rationals) or approximate (reals with rounding) — this matters for the validity of "verified" claims.

## Questions for Authors
1. Can you provide a single end-to-end property that connects the Z3-verified algebraic identities to the system's phase classification output — e.g., "if the Z3 properties hold AND the ODE solver is within tolerance ε, THEN the phase classification is correct with probability p"?
2. What fraction of the 14.3% misclassifications are flagged as low-confidence by the UQ module? If this fraction is below 90%, the self-calibration claim fails.
3. The P20 property appears to verify a statement the implementer themselves found to be false. Can you clarify the exact mathematical claim being verified and why it is correct despite the comment-level analysis suggesting otherwise?
4. Have you considered using abstract interpretation or interval arithmetic over the ODE solver to obtain rigorous error bounds, rather than point-wise Z3 checks on symbolic formulas?
5. What is the formal relationship between the Nyström approximation error and the phase boundary localization error? Is there a theorem or at least a conjectured bound?

## Overall Assessment
PhaseKit presents an ambitious and mathematically interesting system, and the inclusion of Z3 verification is a welcome step toward rigorous ML theory tooling. However, the formal verification component is substantially weaker than presented: it covers isolated algebraic identities rather than system-level guarantees, and at least one property (P20) has visible internal inconsistencies. The gap between "20 verified formulas" and "a verified system" is enormous and inadequately discussed. The moment-closure approximation, which is the system's most significant theoretical limitation, has no formal error bound — the UQ module tracks it heuristically but cannot guarantee soundness. The work would benefit enormously from a clear delineation of what is formally proven versus what is numerically validated versus what is assumed.

**Recommendation:** Weak Reject — The formal verification claims do not withstand scrutiny, and the gap between verified properties and system-level guarantees is not adequately addressed. The mathematical contributions are real but the presentation overstates the rigor of the system.

**Confidence:** 4/5
