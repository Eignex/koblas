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
}

/**
 * The priority every host binding koblas ships registers at. A third-party backend is unprobed, and an ILP64
 * OpenBLAS exports identical symbols while computing wrong answers, caught by reading `openblas_get_config`.
 */
public const val HOST_BACKEND_PRIORITY: Int = 100
