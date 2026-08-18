package com.eignex.koblas.sparse

import com.eignex.koblas.F64SparseVector

internal expect object PlatformSparseVectorKernels : SparseVectorKernels {
    override val name: String

    /** `xᵀ·y` against a dense operand, the one routine a target may accelerate. */
    override fun dot(x: F64SparseVector, y: DoubleArray): Double

    override fun dot(x: F64SparseVector, y: F64SparseVector): Double

    override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector)

    override fun scatter(x: F64SparseVector, out: DoubleArray)

    override fun nrm2(x: F64SparseVector): Double

    override fun asum(x: F64SparseVector): Double
}
