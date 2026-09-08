# Package com.eignex.koblas.dense

Dense linear algebra: the two swappable seams and the routines behind them.

- [F64Kernels] — the level-1 kernels (`dot`, `axpy`, `scale`, `nrm2`, `asum`). These are specialized
  at compile time and replaced as a unit by a selected registered backend; offered through
  [com.eignex.koblas.registerBackend], forced by installing a
  [com.eignex.koblas.F64Context].
- [F64Blas] — the level-2 and level-3 routines in full BLAS alpha/beta/transpose form, plus the triangular
  solves [trsv] / [trsm] and their multiply counterparts. Named Boolean parameters select the triangle,
  transpose, diagonal, and side.
- [F64Blas] providers are offered through [com.eignex.koblas.registerBackend], forced with
  [com.eignex.koblas.installBackends], and resolved as [com.eignex.koblas.koblas]. [F64ReferenceBlas]
  is the portable implementation every backend is validated against.
- Ergonomic entry points cover Kotlin arithmetic operators, matrix products, symmetric updates, and triangular
  solve and multiply operations.

[F64StridedMatrixView][com.eignex.koblas.core.F64StridedMatrixView] and
[F64StridedVectorView][com.eignex.koblas.core.F64StridedVectorView] are live zero-copy views. Panels retain
their parent's column-major leading dimension; their [row][com.eignex.koblas.core.F64StridedMatrixView.row]
and [column][com.eignex.koblas.core.F64StridedMatrixView.column] views retain the corresponding stride.
Mutations through a view or another reference to its backing array are visible to each other. View `gemv` and
`gemm` preserve offsets and strides through JVM and Kotlin/Native CBLAS; negative
vector strides use the portable loop. Output views may share a buffer with disjoint inputs, but an actual
overlap is rejected before mutation because BLAS does not define input/output aliasing for these routines.
