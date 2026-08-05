# Module koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

koblas provides read-only matrix/vector containers with sealed dense and sparse backings —
[MatrixView][com.eignex.koblas.MatrixView] / [DenseMatrix][com.eignex.koblas.DenseMatrix] and
[VectorView][com.eignex.koblas.VectorView] / [DenseVector][com.eignex.koblas.DenseVector] /
[SparseVector][com.eignex.koblas.SparseVector], all `@Serializable` so snapshots round-trip through
`kotlinx.serialization` with their concrete storage preserved.

Light arithmetic lives as free functions over the views: BLAS-1/2 (`dot`, `axpy`, `scale`, `ger`,
`gemv`, `forEachStored`) and an SPD suite (`cholesky`, `solveSpd`,
`invertSpd`). Their inner loops route through an `expect`/`actual` primitive seam that uses SIMD
(`jdk.incubator.vector`) on the JVM and scalar loops elsewhere.

Sparse linear algebra is a first-class peer: a CSC [SparseMatrix][com.eignex.koblas.SparseMatrix] with
matrix–vector products, a Markowitz-pivoting [SparseLu][com.eignex.koblas.sparse.SparseLu] factorization with
`O(nnz)` FTRAN/BTRAN solves, and an [EtaBasis][com.eignex.koblas.sparse.EtaBasis] product-form-of-the-inverse
for `O(m)` rank-1 basis updates between refactorizations — the kernels a sparse simplex or Newton solver
builds on.

The heavier level-2/3 and factorization work — [gemv][com.eignex.koblas.dense.LinearAlgebra.gemv],
[gemm][com.eignex.koblas.dense.LinearAlgebra.gemm] and a general LU
[factor][com.eignex.koblas.dense.LinearAlgebra.factor] / [solve][com.eignex.koblas.dense.LinearAlgebra.solve] — sits
behind the runtime-swappable [LinearAlgebra][com.eignex.koblas.dense.LinearAlgebra] backend so a native
BLAS/LAPACK implementation can replace it without changing callers. [koblas][com.eignex.koblas.koblas]
resolves to an [installBackends][com.eignex.koblas.installBackends] override when set, else
the platform backend when present, else the pure-Kotlin
[ReferenceLinearAlgebra][com.eignex.koblas.dense.ReferenceLinearAlgebra]. Ergonomic entry points
[lu][com.eignex.koblas.dense.lu] / [LuDecomposition.solve][com.eignex.koblas.dense.solve] / [matMul][com.eignex.koblas.dense.matMul]
delegate to the active backend.
