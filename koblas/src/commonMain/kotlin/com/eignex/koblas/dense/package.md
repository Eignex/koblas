# Package com.eignex.koblas.dense

Dense linear algebra: the three swappable seams and the routines behind them.

- [VectorKernels] — the level-1 kernels (`dot`, `axpy`, `scale`, `nrm2`, `asum`). These are specialized
  at compile time rather than dispatched per call, and consult a registered backend only above
  [com.eignex.koblas.DispatchThresholds.level1]; offered through [com.eignex.koblas.registerBackend], forced by installing a
  [com.eignex.koblas.KoblasContext].
- [Blas] — the level-2 and level-3 routines in full BLAS alpha/beta/transpose form, plus the triangular
  solves [trsv] / [trsm] and their multiply counterparts.
- [Lapack] — the factorizations and the solves built on them: LU ([Lapack.factor] / [Lapack.solve],
  with [LuDecomposition] and [determinant]), symmetric indefinite [LdlDecomposition], [QrDecomposition]
  with the least-squares and minimum-norm solves, [PivotedQrDecomposition] when the rank is the question
  rather than the solve, the condition estimate, and the SPD suite
  [cholesky] and the [CholeskyDecomposition] solve and invert.
- [LinearAlgebra] pairs [Blas] and [Lapack]; the two are ranked and selected independently, so a host
  providing one library and not the other still accelerates what it can. Offered through
  [com.eignex.koblas.registerBackend], forced with [com.eignex.koblas.installBackends], resolved as
  [com.eignex.koblas.koblas]. [ReferenceLinearAlgebra]
  is the portable implementation every backend is validated against.
- Ergonomic entry points: [lu], [matMul].
