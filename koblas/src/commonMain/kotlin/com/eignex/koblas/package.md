# Package com.eignex.koblas

The containers every part of koblas speaks, and the free-function arithmetic over them. The routines
themselves live one package down, split by storage: `com.eignex.koblas.dense` and
`com.eignex.koblas.sparse`. See the README's "BLAS coverage" table for the routine-by-routine mapping to
BLAS/LAPACK and the deliberate deviations.

- Containers: [F64MatrixView][com.eignex.koblas.core.F64MatrixView] / [F64DenseMatrix][com.eignex.koblas.core.F64DenseMatrix]
  and [F64VectorView][com.eignex.koblas.core.F64VectorView] / [F64DenseVector][com.eignex.koblas.core.F64DenseVector] /
  [F64SparseVector][com.eignex.koblas.core.F64SparseVector], all `@Serializable`, plus the CSC
  [F64SparseMatrix][com.eignex.koblas.core.F64SparseMatrix]. The view roots are sealed, which is what gives the
  concrete storage a closed set and lets a snapshot round-trip with its type preserved — and is why the
  containers stay in one package rather than splitting with the operations that consume them.
- Free-function arithmetic over the views, dispatching dense or sparse by operand type: [dot], [axpy],
  [scale], [norm2], [asum], [iamax], [copy], [swap], [ger], [times], [transpose], [forEachStored], and
  the matrix 1-norm [norm1].
- Shared machinery: [Backend] (what every backend of every tier reports about itself), structured
  [F64Context.status][com.eignex.koblas.status] snapshots, operation-level
  [F64Context.route][com.eignex.koblas.route] diagnostics, the typed [Workspace] buffer pool,
  [AllocationCapability] and strict [AllocationPolicy] contracts, and the [mathBackend] identifier. None of
  these is per element type.
- Explicit solver configuration: [F64ContextBuilder] resolves independent immutable contexts with
  [F64DispatchPolicy] and [F64FallbackPolicy], without changing the process-wide registry.
- The element type in the names, and the unqualified aliases for the double-precision ones, are collected
  in `Precision.kt`; the `dense` and `sparse` packages each have the same file for their own names.
