# Review: PhaseKit: Finite-Width Phase Diagrams

**Reviewer:** Prof. Anya Petrov (Applied Mathematics & Computational Science, ETH Zürich)
**Score: 8/10**
**Expertise Weight: 0.8** (strong alignment with dynamical systems, bifurcation theory, perturbation theory, numerical methods)

---

PhaseKit applies well-established mathematical techniques — perturbation theory, bifurcation analysis, numerical continuation — to a problem of significant current interest: phase transitions in neural network training dynamics.

**Strengths.** The kernel ODE system with 1/N corrections is a clean perturbation expansion. The perturbation hierarchy Θ(t) = Θ⁽⁰⁾ + (1/N)Θ⁽¹⁾(t) + O(1/N²) is standard in structure but novel in its application to convolutional and residual architectures. The formulation of phase boundaries as eigenvalue crossings in the linearized dynamics is mathematically natural — this is a classical bifurcation problem, and the pseudo-arclength continuation methods for tracing phase boundaries in (γ, depth, data-complexity) space are well-established and reliable. The dimensionless coupling parameter γ = η · N^{−(1−a−b)} providing a unified description of the lazy-to-rich transition across parameterizations is elegant. The Nyström approximation for dataset statistics (kernel-task spectral alignment) is a standard but effective tool for reducing O(n²) to O(nm) complexity.

**Weaknesses.** The Gaussian moment closure (κ₄ = 0) is an uncontrolled approximation whose error is model-dependent. For architectures with strong non-Gaussian fluctuations (e.g., near phase boundaries where fluctuations diverge), the closure may produce qualitatively incorrect results. The sensitivity analysis (±50% κ₄ perturbation) provides some information but is not a rigorous error bound. The one-loop vs. two-loop convergence check is the right diagnostic, but if convergence fails at target widths, the system's predictions are unreliable in precisely the most interesting regions (near phase boundaries). The computational trick of working in kernel space rather than parameter space is mathematically elegant but limits the analysis to the leading-order kernel dynamics — it cannot capture phenomena that arise from the full parameter-space geometry (e.g., loss landscape saddles, edge-of-stability dynamics at discrete time steps). The claim of computing phase diagrams "in minutes on CPU" may be optimistic for the full hyperparameter sweep with 20-seed ground-truth training at each point.

**Verdict.** A mathematically competent application of perturbation theory and bifurcation analysis to neural network training dynamics. The derivations for convolutional and residual architectures are genuine mathematical contributions. The main risk is the moment-closure approximation and its interaction with phase boundary prediction accuracy.
