package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.internal.backend.hostDispatchThresholds

/** Numerical and execution policy shared by host and bundled HFactor providers. */
public data class HfactorOptions(
    /** Smallest stored-entry count routed to native factorization. */
    val factorizeMin: Int? = null,
    /** Whether the binding scales rows before factorization. */
    val equilibrate: Boolean = false,
) {
    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
    }
}

/** Policy for one host HFactor backend instance. */
public data class HfactorConfig(
    /** An absolute path to the library, or the platform lookup chain when null. */
    val libraryPath: String? = null,
    /** Smallest stored-entry count routed to the native factorization; null keeps the platform default. */
    val factorizeMin: Int? = null,
    /** Whether to scale rows before factorizing and undo it in the solves. */
    val equilibrate: Boolean = false,
) {
    /** Creates a deployment-discovered HFactor configuration from shared [options]. */
    public constructor(options: HfactorOptions) : this(null, options)

    /** Creates an HFactor configuration from a library location and shared [options]. */
    public constructor(libraryPath: String?, options: HfactorOptions) : this(
        libraryPath,
        options.factorizeMin,
        options.equilibrate,
    )

    /** Numerical and execution policy, independent of [libraryPath]. */
    public val options: HfactorOptions get() = HfactorOptions(factorizeMin, equilibrate)

    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
    }
}

internal fun HfactorOptions.metadataOptions(): Map<String, String> = mapOf(
    "factorizeMin" to hostDispatchThresholds(factorize = factorizeMin).factorize.toString(),
    "equilibrate" to equilibrate.toString(),
)

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
