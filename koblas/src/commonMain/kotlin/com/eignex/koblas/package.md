# Package com.eignex.koblas

Owning dense containers, borrowed strided vectors, validated CSC sparse containers, the immutable
[KoblasEngine], the reusable [Workspace] routines take their scratch from, and free-function arithmetic over
the matrix and vector contracts.

The default [koblas] engine is selected once and cannot be replaced; the module page describes what each
platform selects. What a call executed is a separate question from what was selected, and these answer it:
[KoblasEngine.explain] for a Level 1 call of a given operation, length and spacing,
[KoblasEngine.denseRouteOf] and [com.eignex.koblas.sparse.SparseBlas.matrixRouteOf] for a matrix call
including the execution grouping a panel was scheduled with, and [KoblasEngine.sparseImplementation] for the
component that owns sparse traversal. Naming a different implementation is for measuring one against
another, so [BuiltinEngines] sits behind [KoblasEngineApi].

High-level operations are extensions in this package, so ordinary use needs only `com.eignex.koblas.*`. The
BLAS contracts live in `com.eignex.koblas.dense`, `com.eignex.koblas.sparse` and `com.eignex.koblas.vendor`;
the sparse primitives and the Level 1 kernels retain caller-owned storage and allocate nothing. [Matrix.gemm]
and [Matrix.gemmInto] dispatch on runtime storage, so dense and sparse operands are usable on either side
without a cast and without densifying either one. Factorization and basis solving are not part of this library;
a consumer that needs them owns its own factors on top of these kernels.
