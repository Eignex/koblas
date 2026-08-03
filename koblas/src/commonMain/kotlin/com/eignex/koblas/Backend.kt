package com.eignex.koblas

/**
 * What every backend reports about itself, whichever half of the seam it implements.
 *
 * A backend may implement [Blas], [Lapack] or both: they are ranked and selected independently, so a
 * host that provides one library and not the other still accelerates what it can.
 */
interface Backend {
    /** A short backend identifier for diagnostics (e.g. `"reference"`). */
    val name: String

    /**
     * Relative preference among simultaneously available backends: automatic selection — JVM
     * classpath discovery and native [registerLinearAlgebra] — picks the highest. The portable
     * reference is 0; native-accelerated backends rank above it (koblas-openblas 100, koblas-cblas
     * 90).
     */
    val priority: Int get() = 0
}
