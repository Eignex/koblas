package com.eignex.koblas.sparse.host.cholmod

/**
 * Policy for one CHOLMOD binding.
 *
 * CHOLMOD is not a backend of its own: it fills the Cholesky routine of a SuiteSparse backend whose LU comes
 * from KLU or UMFPACK, so that the seam resolves to one library collection answering both natively rather
 * than to two backends each declining the other's routine. [searchDirectory] is how that works without a
 * second configured path: a SuiteSparse binding names the directory it loaded its own library from, and
 * CHOLMOD ships beside it.
 */
public data class CholmodConfig(
    /** An absolute CHOLMOD library path, or the platform lookup chain when null. */
    val libraryPath: String? = null,
    /** A directory to look in before the platform chain, where a sibling library was found. */
    val searchDirectory: String? = null,
    /** Smallest stored-entry count routed to the native factorization; null keeps the platform default. */
    val factorizeMin: Int? = null,
) {
    init {
        require(factorizeMin == null || factorizeMin >= 0) { "factorizeMin must not be negative" }
    }
}
