package com.eignex.koblas.internal.dense

import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.configuration.CombinedVectorKernels

/** The implementation a call reaches; kernels that never dispatch answer with their own [DenseVectorKernels.name]. */
internal fun DenseVectorKernels.implementationFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): String? = if (this is CombinedVectorKernels) implementationFor(operation, length, contiguous) else name
