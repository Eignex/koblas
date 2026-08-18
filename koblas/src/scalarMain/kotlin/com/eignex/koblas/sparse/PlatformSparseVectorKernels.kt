package com.eignex.koblas.sparse

import com.eignex.koblas.BackendNames
import com.eignex.koblas.F64SparseVector

/** The portable sparse level-1 kernels, which come from the reference backend. */
internal actual object PlatformSparseVectorKernels : SparseVectorKernels {
    actual override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** The portable version, spelled out because an `expect object` member needs an `actual`. */
    actual override fun dot(x: F64SparseVector, y: DoubleArray): Double = ReferenceSparseLinearAlgebra.dot(x, y)

    // The routines this backend does not provide run the portable versions. Forwarded explicitly rather than
    // by class delegation, which would route a caller's convenience overloads to the portable routine instead
    // of this one, since a delegated member calls back into the delegate.
    actual override fun dot(x: F64SparseVector, y: F64SparseVector): Double = ReferenceSparseLinearAlgebra.dot(x, y)

    actual override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector): Unit =
        ReferenceSparseLinearAlgebra.axpy(y, alpha, x)

    actual override fun scatter(x: F64SparseVector, out: DoubleArray): Unit =
        ReferenceSparseLinearAlgebra.scatter(x, out)

    actual override fun nrm2(x: F64SparseVector): Double = ReferenceSparseLinearAlgebra.nrm2(x)

    actual override fun asum(x: F64SparseVector): Double = ReferenceSparseLinearAlgebra.asum(x)
}
