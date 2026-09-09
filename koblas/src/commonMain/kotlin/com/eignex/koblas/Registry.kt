package com.eignex.koblas

import com.eignex.koblas.dense.PlatformKernels
import com.eignex.koblas.sparse.PlatformSparseKernels

/** The immutable platform-selected BLAS engine used by top-level convenience operations. */
public val koblas: KoblasContext = KoblasContext(PlatformKernels, PlatformSparseKernels)

/** What this runtime selected, for startup logging and benchmark attribution. */
public val koblasInfo: String
    get() = "engine=${koblas.name}"

internal val platformKernels get() = PlatformKernels
