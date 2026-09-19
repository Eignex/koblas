# Package com.eignex.koblas

Owning dense containers, borrowed strided vectors, validated CSC sparse containers, the immutable
[KoblasEngine], and free-function arithmetic over the matrix and vector contracts.

The default [koblas] engine is selected once and cannot be replaced. It binds the Level 1 arm this platform
prefers to portable dense Level 2 and 3. [KoblasEngine.explain] names the Level 1 implementation a call of a
given operation, length and spacing actually reaches; [KoblasEngine.denseImplementation] names the current
dense component. Installed host bindings remain explicit alternatives with binding-derived attribution.
Naming a different implementation is for measuring one against another, so [BuiltinEngines] sits behind
[KoblasEngineApi].

High-level operations are extensions in this package, so ordinary use needs only `com.eignex.koblas.*`. The
BLAS contracts live in `com.eignex.koblas.dense`, `com.eignex.koblas.sparse` and `com.eignex.koblas.vendor`;
the sparse primitives and the Level 1 kernels retain caller-owned storage and allocate nothing. Factorization
and basis solving are not part of this library; a consumer that needs them owns its own factors on top of
these kernels.
