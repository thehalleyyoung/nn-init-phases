# Review: Finite-Width Phase Diagrams for Neural Network Training Dynamics

**Reviewer:** Joseph S. Chang (Automated Reasoning and Logic Expert)
**Score: 6/10**

---

## Summary

This system computes phase diagrams for neural network training dynamics by deriving and numerically solving 1/N-corrected kernel evolution ODEs, integrating multiple theoretical backends via a compiler pipeline. From an automated reasoning and proof-theoretic perspective, the key question is whether the mathematical claims are rigorously established and whether the computational architecture produces certifiable results.

## Strengths

The phase boundary computation as an eigenvalue-crossing bifurcation problem is a well-posed mathematical formulation amenable to rigorous analysis. The transcritical bifurcation characterization — finding γ_c where max Re(λ_i(J(γ_c))) = 0 — reduces phase classification to a concrete decision problem over parametric matrix spectra, connecting training dynamics to classical dynamical systems theory.

The perturbative hierarchy, where the 1/N expansion couples each order to the next cumulant with the µP coupling γ = η · N^{-(1−a−b)} determining convergence, provides a clean criterion for expansion validity.

The retrodiction validation against analytically solved cases (2-layer ReLU MLPs, linear networks) serves as a proof of correctness for the computational machinery.

## Weaknesses

The project makes no attempt at formal verification of its mathematical derivations. The 1/N corrections for convolutional and residual architectures are described as "genuinely new" mathematical contributions, yet they are implemented purely in numerical code without any mechanized proof. The Hessian-Jacobian contraction tensor H_{ijk} for weight-shared architectures involves multi-indexed tensor algebra where index errors are notoriously difficult to detect by testing alone. A Lean or Coq formalization of even the MLP case would substantially increase confidence.

The proof architecture for the convergence claims is incomplete. The claim that one-loop corrections are sufficient is validated only by comparison with two-loop results at discrete width values, but no formal convergence proof is provided. The 1/N expansion may be asymptotic (divergent but with useful partial sums), and the system provides no certificate distinguishing convergent from asymptotic behavior. This is a fundamental gap: without a convergence proof, the system's predictions at any finite width lack theoretical justification.

The algorithmic complexity of the full pipeline is not analyzed. The phase diagram mapper performs adaptive boundary refinement over a multi-dimensional hyperparameter space, but the computational complexity as a function of the number of hyperparameter axes, desired boundary resolution, and ODE stiffness is not characterized. Without complexity bounds, the feasibility claims (minutes to tens of minutes) lack theoretical backing.

The moment-closure truncation lacks a formal error certificate. The ±50% sensitivity analysis on κ₄ provides numerical evidence but not a proof-theoretic guarantee. In automated reasoning, we would require a formal bound: given |κ₄| ≤ B, the phase boundary displacement is at most δ(B). No such bound is derived.

The system generates predictions but no proof certificates. A verifier receiving a phase diagram has no machine-checkable evidence that the underlying ODE was solved correctly, that the eigenvalue computation was accurate, or that the bifurcation detection was sound.

## Recommendations

The project would benefit from: (1) mechanized verification of the H_{ijk} derivation for at least the MLP case, (2) formal convergence analysis distinguishing convergent from asymptotic regimes of the 1/N expansion, (3) complexity analysis of the phase diagram computation, and (4) generation of proof certificates accompanying predictions. Without these, the system is a sophisticated numerical tool but falls short of the formal rigor expected for mathematical contributions claiming novelty.
