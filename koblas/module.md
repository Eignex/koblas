# Module koblas

Dense and sparse linear algebra for Kotlin Multiplatform, with a pluggable BLAS/LAPACK backend seam.

koblas exposes a small [LinearAlgebra][com.eignex.koblas.LinearAlgebra] operation set — BLAS-1
(`dot`/`axpy`/`scal`), BLAS-2/3 (`gemv`/`gemm`) and an LU `factor`/`solve` pair — over flat, row-major
[Matrix][com.eignex.koblas.Matrix] and [Vector][com.eignex.koblas.Vector] buffers. The default backend,
[koblas][com.eignex.koblas.koblas], is a portable pure-Kotlin reference; a platform may supply a native
BLAS/LAPACK-backed backend through [platformLinearAlgebra][com.eignex.koblas.platformLinearAlgebra], and
consumers depend only on the interface.

The op set is intentionally minimal — the pieces a dense linear solver needs — and grows only as
consumers demand.
