package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.internal.backend.hostDispatchThresholds

/** Numerical and execution policy shared by host and bundled BASICLU providers. */
public data class BasicluOptions(
    /** Smallest stored-entry count routed to native factorization. */
    val factorizeMin: Int? = null,
    /** Whether the binding scales rows before factorization. */
    val equilibrate: Boolean = false,
) {
    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
    }
}

/** Policy for one host BASICLU backend instance. */
public data class BasicluConfig(
    /** An absolute BASICLU library path, or the platform lookup chain when null. */
    val libraryPath: String? = null,
    /** Smallest stored-entry count routed to the native factorization; null keeps the platform default. */
    val factorizeMin: Int? = null,
    /** Whether to scale rows before factorizing and undo it in the solves. */
    val equilibrate: Boolean = false,
) {
    /** Creates a deployment-discovered BASICLU configuration from shared [options]. */
    public constructor(options: BasicluOptions) : this(null, options)

    /** Creates a BASICLU configuration from a library location and shared [options]. */
    public constructor(libraryPath: String?, options: BasicluOptions) : this(
        libraryPath,
        options.factorizeMin,
        options.equilibrate,
    )

    /** Numerical and execution policy, independent of [libraryPath]. */
    public val options: BasicluOptions get() = BasicluOptions(factorizeMin, equilibrate)

    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
    }
}

internal fun BasicluOptions.metadataOptions(): Map<String, String> = mapOf(
    "factorizeMin" to hostDispatchThresholds(factorize = factorizeMin).factorize.toString(),
    "equilibrate" to equilibrate.toString(),
)

/** Names used by the platform loader to locate a host BASICLU. */
internal val BASICLU_SONAMES: List<String> = listOf(
    "libbasiclu.so.2",
    "libbasiclu.so",
    "libbasiclu.2.dylib",
    "libbasiclu.dylib",
)

/**
 * Positions of BASICLU's `xstore` this binding reads, from its own header. The store is doubles throughout,
 * counts included, which is what lets one array carry both parameters and statistics.
 */
internal object BasicluStore {
    const val LNZ = 76
    const val UNZ = 77
    const val MIN_PIVOT = 79
    const val MAX_PIVOT = 80
}

/** The status codes this binding distinguishes; every other value is a failure it refactorizes past. */
internal object BasicluStatus {
    const val OK = 0L
    const val SINGULAR_MATRIX = 2L
}
