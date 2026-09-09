# Package com.eignex.koblas

The containers every part of koblas speaks, and the free-function arithmetic over them. The routines
themselves live one package down, split by storage: `com.eignex.koblas.dense` and
`com.eignex.koblas.sparse`. See the README's "Numerical routine coverage" table for the routine-by-routine mapping to
BLAS, sparse operations, and the deliberate deviations.

- Containers: [MatrixStorage][com.eignex.koblas.MatrixStorage] / [DenseMatrix][com.eignex.koblas.DenseMatrix]
  and [VectorStorage][com.eignex.koblas.VectorStorage] / [DenseVector][com.eignex.koblas.DenseVector] /
  [SparseVector][com.eignex.koblas.SparseVector], all `@Serializable`, plus the CSC
  [SparseMatrix][com.eignex.koblas.SparseMatrix]. The storage roots are sealed, which gives the
  concrete storage a closed set and lets a snapshot round-trip with its type preserved — and is why the
  containers stay in one package rather than splitting with the operations that consume them.
- Free-function arithmetic over the read-only contracts, dispatching dense or sparse by operand type: [dot], [axpy],
  [scale], [norm2], [asum], [iamax], [copy], [swap], [ger], [times], the destination-passing
  [gemvInto] and [symvInto], [transpose], [forEachStored], and the matrix 1-norm [norm1].
- Shared machinery: [Backend] (what every backend of every tier reports about itself), structured
  [KoblasContext.status][com.eignex.koblas.status] snapshots, operation-level
  [KoblasContext.route][com.eignex.koblas.route] diagnostics, the typed [Workspace] buffer pool,
  [AllocationCapability] and strict [AllocationPolicy] contracts, and the [mathBackend] identifier. None of
  these is per element type.
- Explicit solver configuration: [ContextBuilder] resolves independent immutable contexts with
  [DispatchPolicy] and [FallbackPolicy], without changing the process-wide registry.
- Koblas currently exposes its single  family directly through these root container names and the
  corresponding backend seams in the `dense` and `sparse` packages.
