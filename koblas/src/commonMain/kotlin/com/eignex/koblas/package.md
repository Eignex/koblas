# Package com.eignex.koblas

Owning and borrowed dense containers, validated CSC sparse containers, workspaces, the immutable
[KoblasContext], and free-function arithmetic over matrix and vector contracts.

The default [koblas] engine is selected once and cannot be replaced. [BuiltinKernels] constructs independent
exact scalar, C, or SIMD engines for tests and benchmarks. [koblasInfo] and [mathBackend] provide concise
read-only implementation attribution. [packedPanels] is the default engine's packed-panel convenience;
explicit engines expose their own bound [KoblasContext.packedPanels].

Dense and sparse BLAS contracts live in `com.eignex.koblas.dense` and `com.eignex.koblas.sparse`. Packed panels
and sparse workspaces retain caller-owned storage and allocation contracts. Factorization and basis-solver APIs
belong to the optional JVM `koblas-hfactor` artifact.
