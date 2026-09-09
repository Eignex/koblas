package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.PlatformVectorKernels
import com.eignex.koblas.internal.configuration.ImplementationNames

/** Scalar sparse leaves used only when cross-compiling a Native publication for a foreign host. */
internal actual object PlatformSparseKernels : SparseKernels by ScalarSparseKernels {
    actual override val name: String get() = ImplementationNames.SCALAR

    actual override fun dot(x: SparseVector, y: DoubleArray): Double = ScalarSparseKernels.dot(x, y)

    actual override fun dot(x: SparseVector, y: SparseVector): Double = ScalarSparseKernels.dot(x, y)

    actual override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector): Unit =
        ScalarSparseKernels.axpy(y, alpha, x)

    actual override fun scatter(x: SparseVector, out: DoubleArray): Unit = ScalarSparseKernels.scatter(x, out)

    actual override fun gather(x: SparseVector, from: DoubleArray): Unit = ScalarSparseKernels.gather(x, from)

    actual override fun gatherZero(x: SparseVector, from: DoubleArray): Unit = ScalarSparseKernels.gatherZero(x, from)

    actual override fun nrm2(x: SparseVector): Double = PlatformVectorKernels.nrm2(x.values, 0, x.values.size)

    actual override fun asum(x: SparseVector): Double = PlatformVectorKernels.asum(x.values, 0, x.values.size)
}
