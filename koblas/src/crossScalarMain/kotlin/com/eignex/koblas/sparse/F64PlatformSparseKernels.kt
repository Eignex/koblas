package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.internal.backend.BackendNames

/** Scalar sparse leaves used only when cross-compiling a Native publication for a foreign host. */
internal actual object F64PlatformSparseKernels : SparseKernels {
    actual override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    actual override fun dot(x: SparseVector, y: DoubleArray): Double = F64ReferenceSparseLinearAlgebra.dot(x, y)

    actual override fun dot(x: SparseVector, y: SparseVector): Double = F64ReferenceSparseLinearAlgebra.dot(x, y)

    actual override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector): Unit =
        F64ReferenceSparseLinearAlgebra.axpy(y, alpha, x)

    actual override fun scatter(x: SparseVector, out: DoubleArray): Unit =
        F64ReferenceSparseLinearAlgebra.scatter(x, out)

    actual override fun gather(x: SparseVector, from: DoubleArray): Unit =
        F64ReferenceSparseLinearAlgebra.gather(x, from)

    actual override fun gatherZero(x: SparseVector, from: DoubleArray): Unit =
        F64ReferenceSparseLinearAlgebra.gatherZero(x, from)

    actual override fun nrm2(x: SparseVector): Double = F64PlatformKernels.nrm2(x.values, 0, x.values.size)

    actual override fun asum(x: SparseVector): Double = F64PlatformKernels.asum(x.values, 0, x.values.size)
}
