package com.eignex.koblas.sparse.host.hfactor

/** Policy for one host HFactor backend instance. */
public data class HfactorConfig(
    /** An absolute path to the library, or the platform lookup chain when null. */
    val libraryPath: String? = null,
    /** Smallest stored-entry count routed to the native factorization; null keeps the platform default. */
    val factorizeMin: Int? = null,
    /** Whether to scale rows before factorizing and undo it in the solves. */
    val equilibrate: Boolean = false,
) {
    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
    }
}

/**
 * Names the platform loader looks for. HFactor is a C++ class inside HiGHS rather than a library with an
 * API of its own, so what is looked for is koblas's own build of it and not a host HiGHS: no distribution
 * ships these entry points.
 */
internal val HFACTOR_SONAMES: List<String> = listOf(
    "libkoblas_hfactor.so.1",
    "libkoblas_hfactor.so",
    "libkoblas_hfactor.1.dylib",
    "libkoblas_hfactor.dylib",
)

/** What the bridge's `koblas_hfactor_update` answers with, which is advice rather than a status. */
internal object HfactorUpdate {
    const val REFUSED = -1
    const val APPLIED = 0
    const val REFACTORIZE = 1
}
