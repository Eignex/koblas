package com.eignex.koblas.sparse.host.basiclu

/** Policy for one host BASICLU backend instance. */
public data class BasicluConfig(
    /** An absolute BASICLU library path, or the platform lookup chain when null. */
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
