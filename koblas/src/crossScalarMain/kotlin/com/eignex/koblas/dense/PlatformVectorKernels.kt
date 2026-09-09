package com.eignex.koblas.dense

internal actual val platformDenseKernelFamilies: DenseKernelFamilies = scalarDenseKernelFamilies

/** Scalar vector family used while cross-compiling for a foreign Native host. */
internal actual object PlatformVectorKernels : DenseVectorKernels by ScalarKernels
