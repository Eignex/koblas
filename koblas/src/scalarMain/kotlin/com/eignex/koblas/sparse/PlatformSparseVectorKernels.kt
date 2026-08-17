package com.eignex.koblas.sparse

import com.eignex.koblas.BackendNames
import com.eignex.koblas.SparseVector

/** The portable sparse level-1 kernels, which are the interface defaults. */
internal actual object PlatformSparseVectorKernels : SparseVectorKernels {
    actual override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** The interface default, spelled out because an `expect object` member needs an `actual`. */
    actual override fun dot(x: SparseVector, y: DoubleArray): Double = super.dot(x, y)
}
