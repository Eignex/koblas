# Module koblas

Dense and sparse BLAS for Kotlin Multiplatform.

Koblas provides mutable owning `Double` containers, dense vectors in either spacing, and validated CSC sparse
storage. Dense matrices are column-major.

Levels 1 to 3 are common Kotlin for both dense and sparse operands, so ordinary matrix computation needs no
installed numerical library. Dense Level 2 and 3 scheduling is shared and portable; the arithmetic inside a
window is a panel, a product block or a diagonal substitution, and which bodies a machine uses is chosen
locally.

Raw indexed sparse kernels operate on caller-owned slices without temporary storage. Stateless structural
helpers for accumulation, touched support, and checked arithmetic are available through
[SparsePrimitives][com.eignex.koblas.sparse.SparsePrimitives]; every buffer they write through is one the
caller passed in, so nothing here allocates behind a hot loop. Routines that need scratch of their own take a
[Workspace][com.eignex.koblas.Workspace], which lends by exact length and retains a bounded number of
lengths, so a repeated call over one shape allocates nothing.

[koblas][com.eignex.koblas.koblas] is an immutable engine selected once for the platform. On the JVM it is
the Vector API kernels this library owns where the incubator module resolved and the portable ones where it
did not. On Kotlin/Native, which has no Vector API, it is the portable schedule plus an installed host
library where one is present, both for Level 1 above its measured widths and for a whole dense Level 2 or 3
call with enough arithmetic to pay for reaching it. Neither platform depends on a library being there: a host
with no supported one computes every level in common Kotlin. Bindings are also explicitly callable on their
own, and that seam has no portable fallback by design, so it raises
[MissingVendorException][com.eignex.koblas.vendor.MissingVendorException] where no supported library is
present rather than substituting this library's arithmetic under a vendor's name.

Every part of the library says what a call actually reached rather than what was selected.
[explain][com.eignex.koblas.KoblasEngine.explain] names the Level 1 component for a given operation, length
and spacing, [denseRouteOf][com.eignex.koblas.KoblasEngine.denseRouteOf] names what a dense matrix call
executes including the bodies its windows reach and the grouping the backend recommended,
[matrixRouteOf][com.eignex.koblas.sparse.SparseBlas.matrixRouteOf] does the same for a sparse matrix call,
[routeOf][com.eignex.koblas.sparse.SparseKernels.routeOf] for sparse Level 1, and
[Blas.routeOf][com.eignex.koblas.vendor.Blas.routeOf] describes one concrete vendor call including
whether it was direct or composed. A selection that falls back is not evidence that its own kernel ran.

Naming an implementation other than the selected one is for measuring the two against each other, so
[BuiltinEngines][com.eignex.koblas.BuiltinEngines] sits behind
[KoblasEngineApi][com.eignex.koblas.KoblasEngineApi] and production code uses
[koblas][com.eignex.koblas.koblas]. The portable kernels are not an alternative to the vectorised ones at a
given size: the vectorised ones already fall back to them below their lane width and for any strided run.
Reaching either way does not change the default engine.

Factorization and basis solving are outside this artifact: a consumer that needs them owns its own factors on
top of these kernels.
