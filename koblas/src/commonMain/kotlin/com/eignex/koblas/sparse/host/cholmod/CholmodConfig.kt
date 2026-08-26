package com.eignex.koblas.sparse.host.cholmod

/**
 * Policy for one CHOLMOD binding.
 *
 * CHOLMOD fills the sparse BLAS seam directly and also supplies Cholesky and `L·D·Lᵀ` routines to the
 * SuiteSparse LU backends. [searchDirectory] lets those composite backends find CHOLMOD beside the library
 * they loaded without requiring a second configured path.
 *
 * This object locates the library only. Sparse-product routing is configured on `CholmodSparseBlas`, while
 * factorization routing belongs to the SuiteSparse decomposition backend that uses CHOLMOD.
 */
public data class CholmodConfig(
    /** An absolute CHOLMOD library path, or the platform lookup chain when null. */
    val libraryPath: String? = null,
    /** A directory to look in before the platform chain, where a sibling library was found. */
    val searchDirectory: String? = null,
)
