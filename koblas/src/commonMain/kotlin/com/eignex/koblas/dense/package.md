# Package com.eignex.koblas.dense

Dense linear algebra: the three swappable seams and the routines behind them.

- [F64Kernels] — the level-1 kernels (`dot`, `axpy`, `scale`, `nrm2`, `asum`). These are specialized
  at compile time rather than dispatched per call, and consult a registered backend only above
  the level-1 dispatch threshold; offered through [com.eignex.koblas.registerBackend], forced by installing a
  [com.eignex.koblas.F64Context].
- [F64Blas] — the level-2 and level-3 routines in full BLAS alpha/beta/transpose form, plus the triangular
  solves [trsv] / [trsm] and their multiply counterparts. Its binary BLAS flags remain Boolean parameters;
  calls supplying more than one should name them, as in
  `trsm(a, b, lower = true, transpose = false, unitDiag = false, right = false)`.
- [F64Decompositions] — the factorizations and the solves built on them: LU ([F64Decompositions.factor] / [F64Decompositions.solve],
  with [F64LuDecomposition] and [determinant]), symmetric indefinite [F64LdlDecomposition], [F64QrDecomposition]
  with the least-squares and minimum-norm solves, [F64PivotedQrDecomposition] when the rank is the question
  rather than the solve, the condition estimate, and the SPD suite
  [cholesky] and the [F64CholeskyDecomposition] solve and invert.
- [F64LinearAlgebra] pairs [F64Blas] and [F64Decompositions]; the two are ranked and selected independently, so a host
  providing one library and not the other still accelerates what it can. Offered through
  [com.eignex.koblas.registerBackend], forced with [com.eignex.koblas.installBackends], resolved as
  [com.eignex.koblas.koblas]. [F64ReferenceLinearAlgebra]
  is the portable implementation every backend is validated against.
- Ergonomic entry points: [lu], [com.eignex.koblas.times].
