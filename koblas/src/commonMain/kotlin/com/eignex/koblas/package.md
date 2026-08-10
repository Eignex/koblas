# Package com.eignex.koblas

The containers every part of koblas speaks, and the free-function arithmetic over them. The routines
themselves live one package down, split by storage: `com.eignex.koblas.dense` and
`com.eignex.koblas.sparse`. See the README's "BLAS coverage" table for the routine-by-routine mapping to
BLAS/LAPACK and the deliberate deviations.

- Containers: [MatrixView] / [DenseMatrix] and [VectorView] / [DenseVector] / [SparseVector], all
  `@Serializable`, plus the CSC [SparseMatrix]. The view roots are sealed, which is what gives the
  concrete storage a closed set and lets a snapshot round-trip with its type preserved — and is why the
  containers stay in one package rather than splitting with the operations that consume them.
- Free-function arithmetic over the views, dispatching dense or sparse by operand type: [dot], [axpy],
  [scale], [norm2], [asum], [iamax], [copy], [swap], [ger], [matVec], [transpose], [forEachStored], and
  the matrix 1-norm [norm1].
- Shared machinery: [Backend] (what every backend of every tier reports about itself), the [Workspace]
  buffer pool, [DispatchThresholds] and the [mathBackend] identifier.
