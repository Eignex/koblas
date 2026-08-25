package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.internal.backend.BackendNames

/** The portable sparse level-1 kernels, which come from the reference backend. */
internal actual object F64PlatformSparseKernels : F64SparseKernels {
    actual override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** The portable version, spelled out because an `expect object` member needs an `actual`. */
    actual override fun dot(x: F64SparseVector, y: DoubleArray): Double = F64ReferenceSparseLinearAlgebra.dot(x, y)

    // The routines this backend does not provide run the portable versions. Forwarded explicitly rather than
    // by class delegation, which would route a caller's convenience overloads to the portable routine instead
    // of this one, since a delegated member calls back into the delegate.
    actual override fun dot(x: F64SparseVector, y: F64SparseVector): Double = F64ReferenceSparseLinearAlgebra.dot(x, y)

    actual override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector): Unit =
        F64ReferenceSparseLinearAlgebra.axpy(y, alpha, x)

    actual override fun scatter(x: F64SparseVector, out: DoubleArray): Unit =
        F64ReferenceSparseLinearAlgebra.scatter(x, out)

    // As on the JVM: both reduce over the stored values alone, so the dense kernels apply unchanged. Here
    // they are the same scalar loops, so this states the shape rather than changing the arithmetic.
    actual override fun nrm2(x: F64SparseVector): Double = F64PlatformKernels.nrm2(x.values, 0, x.values.size)

    actual override fun asum(x: F64SparseVector): Double = F64PlatformKernels.asum(x.values, 0, x.values.size)
}
