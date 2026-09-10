package com.eignex.koblas

import com.eignex.koblas.dense.platformDenseKernelFamilies
import com.eignex.koblas.sparse.platformSparseKernelFamilies

/** The immutable platform-selected BLAS engine used by top-level convenience operations. */
public val koblas: KoblasContext = KoblasContext(
    platformDenseKernelFamilies,
    platformSparseKernelFamilies,
)
