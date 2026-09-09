package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.PlatformKernels
import com.eignex.koblas.internal.backend.BackendNames

/** Scalar sparse leaves used only when cross-compiling a Native publication for a foreign host. */
internal actual object PlatformSparseKernels : SparseKernels {
    actual override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    actual override fun dot(x: SparseVector, y: DoubleArray): Double = ReferenceSparseLinearAlgebra.dot(x, y)

    actual override fun dot(x: SparseVector, y: SparseVector): Double = ReferenceSparseLinearAlgebra.dot(x, y)

    actual override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector): Unit =
        ReferenceSparseLinearAlgebra.axpy(y, alpha, x)

    actual override fun scatter(x: SparseVector, out: DoubleArray): Unit = ReferenceSparseLinearAlgebra.scatter(x, out)

    actual override fun gather(x: SparseVector, from: DoubleArray): Unit = ReferenceSparseLinearAlgebra.gather(x, from)

    actual override fun gatherZero(x: SparseVector, from: DoubleArray): Unit =
        ReferenceSparseLinearAlgebra.gatherZero(x, from)

    actual override fun nrm2(x: SparseVector): Double = PlatformKernels.nrm2(x.values, 0, x.values.size)

    actual override fun asum(x: SparseVector): Double = PlatformKernels.asum(x.values, 0, x.values.size)
}
