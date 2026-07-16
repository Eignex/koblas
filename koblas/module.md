# Module koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

koblas provides read-only matrix/vector containers with sealed dense and sparse backings —
[MatrixView][com.eignex.koblas.MatrixView] / [DenseMatrix][com.eignex.koblas.DenseMatrix] and
[VectorView][com.eignex.koblas.VectorView] / [DenseVector][com.eignex.koblas.DenseVector] /
[SparseVector][com.eignex.koblas.SparseVector], all `@Serializable` so snapshots round-trip through
`kotlinx.serialization` with their concrete storage preserved.

Light arithmetic lives as free functions over the views: BLAS-1/2 (`dot`, `axpy`, `scale`, `addOuter`,
`matVec`, `forEachStored`) and an SPD suite (`cholesky`, `choleskyDowndateInPlace`, `solveSpd`,
`invertSpd`). Their inner loops route through an `expect`/`actual` primitive seam that uses SIMD
(`jdk.incubator.vector`) on the JVM and scalar loops elsewhere.

The heavier level-2/3 and factorization work — [gemv][com.eignex.koblas.LinearAlgebra.gemv],
[gemm][com.eignex.koblas.LinearAlgebra.gemm] and a general LU
[factor][com.eignex.koblas.LinearAlgebra.factor] / [solve][com.eignex.koblas.LinearAlgebra.solve] — sits
behind the runtime-swappable [LinearAlgebra][com.eignex.koblas.LinearAlgebra] backend so a native
BLAS/LAPACK or GPU implementation can replace it without changing callers. [koblas][com.eignex.koblas.koblas]
resolves to the platform backend when present, else the pure-Kotlin
[ReferenceLinearAlgebra][com.eignex.koblas.ReferenceLinearAlgebra]. Ergonomic entry points
[lu][com.eignex.koblas.lu] / [LuDecomposition.solve][com.eignex.koblas.solve] / [matMul][com.eignex.koblas.matMul]
delegate to the active backend.
