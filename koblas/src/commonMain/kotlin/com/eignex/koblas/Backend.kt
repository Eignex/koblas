package com.eignex.koblas

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64VectorKernels

/** What every backend reports about itself. */
public interface Backend {
    /** A short backend identifier for diagnostics (e.g. `"reference"`). */
    public val name: String

    /**
     * Relative preference among the backends offered for one half ([F64Blas], [F64Lapack], [F64VectorKernels] or a
     * sparse counterpart). [registerBackend] picks the highest; the portable reference is 0.
     */
    public val priority: Int get() = 0

    /**
     * Whether this is koblas's own implementation rather than a binding to a host library. The compiled-in
     * SIMD kernels are portable however fast they are; only something calling out counts as accelerated.
     */
    public val isPortable: Boolean get() = false

    /**
     * Whether this backend can do work on this host. koblas's own implementations always can, so the default
     * is true; a binding reports whether the library it calls resolved.
     *
     * Each half answers for itself, since a host can provide CBLAS without LAPACKE, or OpenBLAS without
     * UMFPACK. Registration does not consult this: koblas registers its UMFPACK binding on a bare library
     * lookup and lets the binding fall back per call, so a registered backend may still report false here.
     * Read it to report what a host offers, or before installing one explicitly.
     */
    public val isAvailable: Boolean get() = true
}

/**
 * The priority every host binding koblas ships registers at. A third-party backend is unprobed, and an ILP64
 * OpenBLAS exports identical symbols while computing wrong answers, caught by reading `openblas_get_config`.
 */
public const val HOST_BACKEND_PRIORITY: Int = 100
