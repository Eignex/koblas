package com.eignex.koblas.sparse.host.cholmod

import java.io.File

/**
 * The CHOLMOD that ships beside a SuiteSparse library already loaded from [libraryPath], for a binding
 * filling the Cholesky half of the seam its LU holds.
 *
 * A bundled artifact extracts the whole collection into one directory, so the sibling is exactly the CHOLMOD
 * that was built with it. A host binding configures no path and falls through to the platform chain, which
 * finds whatever SuiteSparse the machine has.
 */
internal fun suiteSparseCholesky(libraryPath: String?): CholmodCholesky =
    CholmodCholesky(CholmodConfig(searchDirectory = libraryPath?.let { File(it).parent }))
