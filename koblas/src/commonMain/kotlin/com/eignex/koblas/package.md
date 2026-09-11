# Package com.eignex.koblas

Owning and borrowed dense containers, validated CSC sparse containers, workspaces, the immutable
[KoblasEngine], and free-function arithmetic over matrix and vector contracts.

The default [koblas] engine is selected once and cannot be replaced. [BuiltinEngines] constructs independent
exact scalar, C, or SIMD engines for tests and benchmarks. [KoblasEngine.name] provides concise read-only
implementation attribution.

High-level operations are extensions in this package, so ordinary use needs only `com.eignex.koblas.*`.
Dense and sparse BLAS contracts live in `com.eignex.koblas.dense` and `com.eignex.koblas.sparse`; packed panels
and sparse slices retain caller-owned storage and allocation contracts. Factorization and basis-solver APIs
belong to the optional JVM `koblas-hfactor` artifact.
