# Module koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

koblas provides read-only matrix/vector containers with sealed dense and sparse backings —
[MatrixView][com.eignex.koblas.MatrixView] / [DenseMatrix][com.eignex.koblas.DenseMatrix] and
[VectorView][com.eignex.koblas.VectorView] / [DenseVector][com.eignex.koblas.DenseVector] /
[SparseVector][com.eignex.koblas.SparseVector], all `@Serializable` so snapshots round-trip through
`kotlinx.serialization` with their concrete storage preserved.

Arithmetic lives as free functions over the views: BLAS-1/2 (`dot`, `axpy`, `scale`, `addOuter`,
`matVec`, `forEachStored`) and an SPD suite (`cholesky`, `choleskyDowndateInPlace`, `solveSpd`,
`invertSpd`). Dense hot paths route through an `expect`/`actual` primitive seam that uses SIMD
(`jdk.incubator.vector`) on the JVM and scalar loops elsewhere; a tuned BLAS/LAPACK backend can replace
the seam per platform later without changing callers.
