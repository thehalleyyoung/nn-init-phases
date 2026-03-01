# Review: Finite-Width Phase Diagrams for Neural Network Training Dynamics

**Reviewer:** Lila Zhang (Symbolic Reasoning and AI Expert)
**Score: 7/10**

---

## Summary

This project builds a computational framework for predicting neural network training regimes (lazy vs. rich) by symbolically computing finite-width corrections to the Neural Tangent Kernel and integrating multiple theoretical backends into a unified phase diagram mapper. The symbolic kernel engine and tensor program algebra form the core symbolic reasoning components.

## Strengths

The symbolic kernel engine's handling of per-layer NTK and NNGP kernel composition across dense, convolutional, and residual architectures is an impressive piece of symbolic computation. The Hessian-Jacobian contraction tensor H_{ijk} — which decomposes into per-layer contributions scaling as 1/N_ℓ — requires careful symbolic manipulation of multi-indexed tensor expressions with distinct structures for weight-shared (ConvNet) and skip-connected (ResNet) architectures. This is genuinely new symbolic derivation work.

The tensor program algebra, where weight matrices carry scaling exponents (a, b, c) propagated through an architecture DAG via composition rules for MatMul, nonlinearity, aggregation, and skip connections, constitutes a well-defined symbolic rewriting system. The derivation of the effective coupling γ = η · N^{-(1−a−b)} from these propagated exponents is a clean algebraic result connecting symbolic analysis to dynamical predictions. The computation-graph IR that simultaneously supports multi-theory annotation (NTK kernel recursion, µP scaling classes, mean-field depth indexing) is a thoughtful design that avoids the pitfalls of ad-hoc backend-specific representations.

The formulation of phase boundaries as eigenvalue-crossing problems (transcritical bifurcations in the linearized kernel ODE) gives the phase classification a formal algebraic characterization rather than relying on heuristic clustering.

## Weaknesses

The symbolic kernel engine's completeness is limited by the restriction to a fixed set of nonlinearities. The symbolic integration required for convolutional NTK at finite width has "no closed-form solution" per the problem statement, meaning the system falls back to numerical integration. This breaks the symbolic chain: the downstream bifurcation analysis operates on numerically approximated kernels rather than exact symbolic expressions, and error propagation through the eigenvalue computation is not bounded symbolically.

The moment-closure truncation at 4th-order cumulants is a non-symbolic approximation injected into an otherwise algebraic framework. The Gaussian closure ansatz (κ₄ = 0) is an unjustified symbolic simplification — it is not derivable from the preceding symbolic computation but rather imposed as an external assumption. A more principled approach would be to derive the closure condition symbolically from the architecture structure, or at minimum to provide symbolic bounds on the truncation error.

The decidability properties of the phase boundary computation are not analyzed. The bifurcation detection problem — finding γ_c where max Re(λ_i(J(γ_c))) = 0 — involves computing eigenvalues of a parametric matrix family, which is generically decidable for polynomial entries but whose complexity depends on the degree and structure of the entries. The system does not characterize when the symbolic computation is guaranteed to terminate or produce correct results.

The logical soundness of the multi-backend fusion is underspecified. When the NTK backend, mean-field backend, and µP backend produce conflicting regime predictions, the resolution strategy is not formally defined. Without a formal conflict-resolution logic, the fused prediction lacks soundness guarantees.

## Recommendations

The project should (1) establish symbolic error bounds for the numerical kernel integration, (2) characterize the decidability and complexity of the parametric eigenvalue problem underlying bifurcation detection, (3) define a formal logic for multi-backend fusion with proved soundness, and (4) investigate whether architecture-dependent moment-closure conditions can be derived symbolically rather than imposed.
