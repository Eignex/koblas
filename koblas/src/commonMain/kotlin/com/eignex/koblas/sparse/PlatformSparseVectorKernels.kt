package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector

internal expect object PlatformSparseVectorKernels : SparseVectorKernels {
    override val name: String

    /** `xᵀ·y` against a dense operand, the one routine a target may accelerate. */
    override fun dot(x: SparseVector, y: DoubleArray): Double
}
