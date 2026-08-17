package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.VectorKernels

/** What every backend reports about itself. */
public interface Backend {
    /** A short backend identifier for diagnostics (e.g. `"reference"`). */
    public val name: String

    /**
     * Relative preference among the backends offered for one half ([Blas], [Lapack], [VectorKernels] or a
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
     * UMFPACK. Registration already skips an unavailable backend, so this is for a caller installing one
     * explicitly or reporting what a host offers.
     */
    public val isAvailable: Boolean get() = true
}

/**
 * The priority every host binding koblas ships registers at. A third-party backend is unprobed, and an ILP64
 * OpenBLAS exports identical symbols while computing wrong answers, caught by reading `openblas_get_config`.
 */
public const val HOST_BACKEND_PRIORITY: Int = 100
