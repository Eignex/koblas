package com.eignex.koblas.sparse.host.spqr

/**
 * SPQR's sonames and codes; its operands are CHOLMOD structs, so the marshalling is all next door. The `_i_`
 * entry points take the 32-bit indices that package builds, where the unsuffixed ones are `int64_t`.
 */
internal val SPQR_SONAMES: List<String> = listOf(
    "libspqr.so.4",
    "libspqr.so.3",
    "libspqr.so.2",
    "libspqr.so",
    "libspqr.4.dylib",
    "libspqr.dylib",
    "/opt/homebrew/opt/suite-sparse/lib/libspqr.dylib", // Homebrew is keg-only
    "/opt/homebrew/opt/suite-sparse/lib/libspqr.4.dylib",
    "/usr/local/opt/suite-sparse/lib/libspqr.dylib",
)

internal const val SPQR_QTX = 0

internal const val SPQR_QX = 1

internal const val SPQR_RETX_EQUALS_B = 1

internal const val SPQR_DEFAULT_TOL = -2.0

internal const val SPQR_NO_TOL = -1.0

internal const val SPQR_ORDERING_DEFAULT = 7

internal const val SPQR_ORDERING_NATURAL = 1

/** Fill-reducing column orderings SPQR offers. */
public enum class SpqrOrdering {
    /** SPQR's own strategy. */
    DEFAULT,

    /** The order the matrix arrives in. */
    NATURAL,
}

internal fun SpqrOrdering.code(): Int = when (this) {
    SpqrOrdering.DEFAULT -> SPQR_ORDERING_DEFAULT
    SpqrOrdering.NATURAL -> SPQR_ORDERING_NATURAL
}
