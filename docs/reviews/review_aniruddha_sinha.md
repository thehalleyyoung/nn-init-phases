# Review: Finite-Width Phase Diagrams for Neural Network Training Dynamics

**Reviewer:** Aniruddha Sinha (Model Checking and AI-Assisted Verification Expert)
**Score: 6/10**

---

## Summary

This system computes finite-width phase diagrams by integrating NTK theory with 1/N corrections, mean-field limits, and µP scaling into a unified compiler pipeline. From a verification perspective, the key question is whether the system provides reliable correctness guarantees for its regime classifications and phase boundary predictions.

## Strengths

The pipeline architecture — parsing architecture specifications into a computation-graph IR, then lowering through parallel backends — is a well-structured approach that cleanly separates concerns. The multi-backend fusion (symbolic kernel engine, mean-field baseline, tensor program analyzer) with explicit confidence stratification (high/moderate/near-boundary) is analogous to multi-analysis verification frameworks where different abstraction domains contribute complementary information.

The bifurcation detection via eigenvalue tracking of the linearized kernel ODE is a principled approach with clear mathematical semantics. The use of pseudo-arclength continuation along phase boundaries provides numerical stability near bifurcation points where standard parameter continuation would fail. The adaptive boundary refinement in the phase diagram mapper is essential for efficient exploration of multi-dimensional hyperparameter spaces.

The retrodiction validation against analytically known results (Chizat & Bach 2019, Saxe et al. 2014) provides necessary sanity checks that serve a similar role to regression testing in verification pipelines.

## Weaknesses

The system fundamentally lacks formal verification of its core predictions. The regime classification rules — the criteria determining whether a point is "lazy," "rich," "chaotic," or "divergent" — are not formally specified or verified. The classification depends on eigenvalue gap magnitudes and threshold comparisons, but these thresholds appear to be empirically calibrated rather than derived from the mathematical structure. A model-checking approach would formalize the classification criteria as temporal properties over the ODE solution trajectory.

The state-space exploration of the hyperparameter grid suffers from a coverage problem analogous to state explosion. The system sweeps over (learning rate, width, initialization scale, depth, data complexity), creating a combinatorial space that adaptive refinement cannot fully explore. The 100–1000 ODE solves mentioned in the feasibility section are likely insufficient for the claimed multi-dimensional phase diagrams. No formal argument bounds the approximation error introduced by grid spacing.

The mean-field backend's reliance on precomputed solutions for 2–3 layer networks with "empirical depth corrections without theoretical justification for normalized architectures" is a correctness concern. The system claims to compute phase diagrams for deeper architectures, but the mean-field contribution for these architectures rests on unjustified extrapolation. No bound on the resulting error is provided.

The convergence verification (one-loop vs. two-loop agreement) is checked only at discrete width values {64, 128, 256, 512, 1024}. Convergence at N = 256 does not guarantee convergence at N = 300. The system should provide continuous convergence certificates or at minimum interpolation guarantees between checked points.

The exclusion of transformer architectures — stated as a scope limitation — significantly reduces practical impact given that transformers dominate current ML practice. The mathematical obstacle (softmax breaking NTK/mean-field analyses) is genuine but should be acknowledged as a fundamental limitation rather than a scoping choice.

## Recommendations

To strengthen correctness guarantees: (1) formalize the regime classification criteria as verifiable properties, (2) provide coverage guarantees for the hyperparameter grid exploration with bounded approximation error, (3) derive error bounds for the mean-field depth extrapolation, and (4) establish monotonicity or continuity properties of convergence between checked widths. The system would benefit from treating phase boundary accuracy as a verification problem rather than purely a numerical analysis problem.
