# Package com.eignex.koblas.dense

Dense linear algebra: the three swappable seams and the routines behind them.

- [F64Kernels] — the level-1 kernels (`dot`, `axpy`, `scale`, `nrm2`, `asum`). These are specialized
  at compile time and replaced as a unit by a selected registered backend; offered through
  [com.eignex.koblas.registerBackend], forced by installing a
  [com.eignex.koblas.F64Context].
- [F64Blas] — the level-2 and level-3 routines in full BLAS alpha/beta/transpose form, plus the triangular
  solves [trsv] / [trsm] and their multiply counterparts. Named Boolean parameters select the triangle,
  transpose, diagonal, and side.
- [F64Decompositions] — the factorizations and the solves built on them: LU ([F64Decompositions.factor] / [F64Decompositions.solve],
  with [F64LuDecomposition] and [determinant]), Bunch-Kaufman numerically pivoted symmetric-indefinite
  [F64PivotedSymmetricIndefiniteDecomposition] through [F64Decompositions.pivotedSymmetricIndefinite],
  [F64QrDecomposition]
  with the least-squares and minimum-norm solves, [F64PivotedQrDecomposition] when the rank is the question
  rather than the solve, the condition estimate, and the SPD suite
  [cholesky] and the [F64CholeskyDecomposition] solve, invert and in-place
  [F64CholeskyDecomposition.rankUpdate].
- [F64LinearAlgebra] pairs [F64Blas] and [F64Decompositions]; the two are ranked and selected independently, so a host
  providing one library and not the other still accelerates what it can. Offered through
  [com.eignex.koblas.registerBackend], forced with [com.eignex.koblas.installBackends], resolved as
  [com.eignex.koblas.koblas]. [F64ReferenceLinearAlgebra]
  is the portable implementation every backend is validated against.
- Ergonomic entry points: [lu], [pivotedSymmetricIndefinite], and Kotlin arithmetic operators. The dense
  factorization deliberately has a different name from sparse [com.eignex.koblas.sparse.quasiDefiniteLdl]:
  dense Bunch-Kaufman chooses numerical pivots for stability, while sparse quasi-definite LDL preserves its
  fill-reducing ordering.

Dense decomposition objects own only Kotlin arrays. They deliberately are not `AutoCloseable`: there is no native
factor handle or symbolic analysis to release. Retain a factor to solve repeated right-hand sides, and refactor one
in place for a same-sized matrix with `refactorInto` — [F64LuDecomposition.refactorInto],
[F64CholeskyDecomposition.refactorInto], [F64PivotedSymmetricIndefiniteDecomposition.refactorInto], and
[F64QrDecomposition.refactorInto] all
reuse the factorization's existing buffers; dense factorizations have no `prepare` or symbolic-analysis counterpart
because neither adds reusable work beyond those buffers. Every factor-owned operation delegates through the active
[com.eignex.koblas.koblas] context, including the `Into` forms that retain caller-owned destinations and accept a
[com.eignex.koblas.Workspace] where staging is needed.

[F64StridedMatrixView][com.eignex.koblas.core.F64StridedMatrixView] and
[F64StridedVectorView][com.eignex.koblas.core.F64StridedVectorView] are live zero-copy views. Panels retain
their parent's column-major leading dimension; their [row][com.eignex.koblas.core.F64StridedMatrixView.row]
and [column][com.eignex.koblas.core.F64StridedMatrixView.column] views retain the corresponding stride.
Mutations through a view or another reference to its backing array are visible to each other. View `gemv` and
`gemm` preserve offsets and strides through JVM and Kotlin/Native CBLAS; negative
vector strides use the portable loop. Output views may share a buffer with disjoint inputs, but an actual
overlap is rejected before mutation because BLAS does not define input/output aliasing for these routines.
Factorizations continue to accept contiguous matrices: their result objects keep private in-place factor storage,
so a decomposition view would need a separate mutation contract.
