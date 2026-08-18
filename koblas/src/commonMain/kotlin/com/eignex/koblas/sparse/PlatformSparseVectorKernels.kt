package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector

internal expect object PlatformSparseVectorKernels : SparseVectorKernels {
    override val name: String

    /** `xᵀ·y` against a dense operand, the one routine a target may accelerate. */
    override fun dot(x: SparseVector, y: DoubleArray): Double

    override fun dot(x: SparseVector, y: SparseVector): Double

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector)

    override fun scatter(x: SparseVector, out: DoubleArray)

    override fun nrm2(x: SparseVector): Double

    override fun asum(x: SparseVector): Double
}
