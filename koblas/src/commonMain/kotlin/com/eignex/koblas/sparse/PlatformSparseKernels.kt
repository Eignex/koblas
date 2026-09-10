package com.eignex.koblas.sparse

internal expect val platformSparseKernelFamilies: SparseKernelFamilies

internal val PlatformSparseKernels: SparseKernels
    get() = platformSparseKernelFamilies.vector
