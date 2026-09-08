# Module koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

Koblas currently implements one numerical family: F64, with Kotlin `Double` elements. Its public container
and backend names are the concise F64 defaults: [DenseMatrix][com.eignex.koblas.DenseMatrix] and
[DenseVector][com.eignex.koblas.DenseVector], for example. The README's "Data and storage" section says
what an element type added later brings and what stays shared.

Koblas provides mutable owning containers through read-only matrix/vector contracts, with sealed dense and
sparse backings —
[MatrixStorage][com.eignex.koblas.MatrixStorage] / [DenseMatrix][com.eignex.koblas.DenseMatrix] and
[VectorStorage][com.eignex.koblas.VectorStorage] / [DenseVector][com.eignex.koblas.DenseVector] /
[SparseVector][com.eignex.koblas.SparseVector], all `@Serializable` so snapshots round-trip through
`kotlinx.serialization` with their concrete storage preserved.

Light arithmetic lives as free functions over the views: BLAS-1/2 (`dot`, `axpy`, `scale`, `ger`,
`gemv`, `forEachStored`). Their inner loops route through an `expect`/`actual` primitive seam that uses SIMD
(`jdk.incubator.vector`) on the JVM when present and compiled C kernels on Native and non-SIMD JVMs.

Sparse linear algebra is a first-class peer: a CSC [SparseMatrix][com.eignex.koblas.SparseMatrix]
with matrix-vector and matrix-matrix products, general and repeated-pattern LU, Cholesky, quasi-definite LDL,
and distinct simplex-basis capabilities. Sparse factors provide vector and
block solves, deterministic lifecycle, allocation contracts, and typed factor access. The README's "Numerical
routine coverage" and "Sparse workflows" sections map these semantic roles to portable and native providers.

The level-2/3 dense work sits behind the runtime-swappable
[Blas][com.eignex.koblas.dense.Blas] backend so a native BLAS implementation can replace it without
changing callers. [koblas][com.eignex.koblas.koblas]
resolves to an [installBackends][com.eignex.koblas.installBackends] override when set, else
the platform backend when present, else the pure-Kotlin
[F64ReferenceBlas][com.eignex.koblas.dense.F64ReferenceBlas]. Dense matrix products and triangular operations
delegate to the active backend.
