package com.eignex.koblas.sparse.host.cholmod

/**
 * Policy for one CHOLMOD binding.
 *
 * CHOLMOD is not a backend of its own: it fills the Cholesky and `L·D·Lᵀ` routines of a SuiteSparse backend
 * whose LU comes from KLU or UMFPACK, so that the seam resolves to one library collection answering all of
 * them natively rather than to two backends each declining the other's routine. [searchDirectory] is how that
 * works without a second configured path: a SuiteSparse binding names the directory it loaded its own library
 * from, and CHOLMOD ships beside it.
 *
 * No size gate here. What decides between this and the portable factorization is the gate on
 * [com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter], which runs before the backend reaches this
 * at all, and a second one here could only disagree with it.
 */
public data class CholmodConfig(
    /** An absolute CHOLMOD library path, or the platform lookup chain when null. */
    val libraryPath: String? = null,
    /** A directory to look in before the platform chain, where a sibling library was found. */
    val searchDirectory: String? = null,
)
