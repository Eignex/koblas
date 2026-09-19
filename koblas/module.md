# Module koblas

Dense and sparse BLAS for Kotlin Multiplatform.

Koblas provides mutable owning `Double` containers, dense vectors in either spacing, and validated CSC sparse
storage. Dense matrices are column-major.

Raw indexed sparse kernels operate on caller-owned slices without temporary storage. Stateless structural
helpers for accumulation, touched support, and checked arithmetic are available through
[SparsePrimitives][com.eignex.koblas.sparse.SparsePrimitives]; every buffer they write through is one the
caller passed in, so nothing here allocates behind a hot loop.

[koblas][com.eignex.koblas.koblas] is an immutable engine selected once for the platform, and it is two
halves. Level 1 and the sparse primitives are portable Kotlin, preferring the Vector API reductions on the
JVM and scalar Kotlin elsewhere; they keep working on any host. Level 2 and 3 are whole calls to an installed
vendor BLAS and have no portable fallback, so they raise
[MissingVendorException][com.eignex.koblas.vendor.MissingVendorException] where no supported library is
present rather than substituting slower arithmetic under the same name.

Both halves say what a call actually reached rather than what was selected.
[explain][com.eignex.koblas.KoblasEngine.explain] names the Level 1 component for a given operation, length
and spacing, [denseRouteOf][com.eignex.koblas.KoblasEngine.denseRouteOf] names what a dense matrix call
executes including the panel each of its windows reaches and the grouping the backend recommended,
[routeOf][com.eignex.koblas.sparse.SparseKernels.routeOf] does the same for sparse Level 1, and
[Blas.routeOf][com.eignex.koblas.vendor.Blas.routeOf] describes one concrete vendor call including
whether it was direct or composed. A selection that falls back is not evidence that its own kernel ran.

Naming a Level 1 implementation other than the selected one is for measuring the two against each other, so
[BuiltinEngines][com.eignex.koblas.BuiltinEngines] sits behind
[KoblasEngineApi][com.eignex.koblas.KoblasEngineApi] and production code uses
[koblas][com.eignex.koblas.koblas]. The portable kernels are not an alternative to the vectorised ones at a
given size: the vectorised ones already fall back to them below their lane width and for any strided run.
Reaching either way does not change the default engine.

Factorization and basis solving are outside this artifact: a consumer that needs them owns its own factors on
top of these kernels.
