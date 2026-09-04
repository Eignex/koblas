package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.sparse.host.parentDirectory

/** The SPQR that ships beside a SuiteSparse library already loaded from [libraryPath]. */
internal fun suiteSparseQr(libraryPath: String?): SpqrQr =
    SpqrQr(SpqrConfig(searchDirectory = libraryPath?.let(::parentDirectory)))
