package com.eignex.koblas.internal.dense

import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels

/** Dense Level 1 kernels that hand some lengths or layouts to another implementation. */
internal interface DispatchingVectorKernels : DenseVectorKernels {
    /**
     * The implementation a call of this [operation] and [length] reaches, or null when its own values decide,
     * as for a norm that retries through a rescaling loop when the square sum leaves the normal range.
     */
    fun implementationFor(operation: DenseOperation, length: Int, contiguous: Boolean): String?
}

/** The implementation a call reaches; kernels that never dispatch answer with their own [DenseVectorKernels.name]. */
internal fun DenseVectorKernels.implementationFor(
    operation: DenseOperation,
    length: Int,
    contiguous: Boolean = true,
): String? = if (this is DispatchingVectorKernels) implementationFor(operation, length, contiguous) else name
