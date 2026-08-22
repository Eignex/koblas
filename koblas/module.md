# Module koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

Every container and every backend half names its element type. Double precision is the only one
implemented, so the names all begin `F64`, and the unqualified names are aliases for them: `DenseMatrix` is
[F64DenseMatrix][com.eignex.koblas.core.F64DenseMatrix]. The README's "Element types" section says what an
element type added later brings and what stays shared.

Koblas provides mutable owning containers through read-only matrix/vector contracts, with sealed dense and
sparse backings —
[F64MatrixView][com.eignex.koblas.core.F64MatrixView] / [F64DenseMatrix][com.eignex.koblas.core.F64DenseMatrix] and
[F64VectorView][com.eignex.koblas.core.F64VectorView] / [F64DenseVector][com.eignex.koblas.core.F64DenseVector] /
[F64SparseVector][com.eignex.koblas.core.F64SparseVector], all `@Serializable` so snapshots round-trip through
`kotlinx.serialization` with their concrete storage preserved.

Light arithmetic lives as free functions over the views: BLAS-1/2 (`dot`, `axpy`, `scale`, `ger`,
`gemv`, `forEachStored`) and an SPD suite (`cholesky`, `solveSpd`,
`invertSpd`). Their inner loops route through an `expect`/`actual` primitive seam that uses SIMD
(`jdk.incubator.vector`) on the JVM and scalar loops elsewhere.

Sparse linear algebra is a first-class peer: a CSC [F64SparseMatrix][com.eignex.koblas.core.F64SparseMatrix] with
matrix–vector products and a Markowitz-pivoting [F64SparseLuFactorization][com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization] factorization
with `O(nnz)` forward and transposed solves — the kernels a sparse simplex or Newton solver builds on.

The heavier level-2/3 and factorization work — [gemv][com.eignex.koblas.dense.F64LinearAlgebra.gemv],
[gemm][com.eignex.koblas.dense.F64LinearAlgebra.gemm] and a general LU
[factor][com.eignex.koblas.dense.F64LinearAlgebra.factor] / [solve][com.eignex.koblas.dense.F64LinearAlgebra.solve] — sits
behind the runtime-swappable [F64LinearAlgebra][com.eignex.koblas.dense.F64LinearAlgebra] backend so a native
BLAS/LAPACK implementation can replace it without changing callers. [koblas][com.eignex.koblas.koblas]
resolves to an [installBackends][com.eignex.koblas.installBackends] override when set, else
the platform backend when present, else the pure-Kotlin
[F64ReferenceLinearAlgebra][com.eignex.koblas.dense.F64ReferenceLinearAlgebra]. Ergonomic entry points
[lu][com.eignex.koblas.dense.lu] / [F64LuDecomposition.solve][com.eignex.koblas.dense.solve] / [matMul][com.eignex.koblas.dense.matMul]
delegate to the active backend.
