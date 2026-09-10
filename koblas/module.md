# Module koblas

Dense and sparse BLAS for Kotlin Multiplatform.

Koblas provides mutable owning `Double` containers, live strided dense views, validated CSC sparse storage,
caller-owned workspaces, and packed arithmetic helpers. Dense matrices are column-major.

[koblas][com.eignex.koblas.koblas] is an immutable engine selected once for the platform. JVM selection prefers
the Vector API, then Koblas's bundled C kernels, then scalar Kotlin. Kotlin/Native uses the bundled C kernels
with scalar fallbacks. Shared dense and sparse matrix algorithms are bound directly to those selected kernels.

Exact scalar, C, and SIMD engines are available through the experimental
[BuiltinEngines][com.eignex.koblas.BuiltinEngines] construction seam for tests and benchmarks. Constructing one
does not change the default engine. Factorization and basis solving are outside this artifact; the optional JVM
`koblas-hfactor` artifact exposes HFactor directly.
