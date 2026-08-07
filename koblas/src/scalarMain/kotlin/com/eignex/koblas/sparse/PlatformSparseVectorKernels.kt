package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector

/**
 * The portable sparse level-1 kernels, which are the interface defaults.
 *
 * Nothing is overridden. There is no vector API on these targets, and the indexed access pattern is what a
 * host sparse BLAS would have to beat — see the JVM actual for why only one of the four routines gains
 * anything even where wide registers exist.
 */
internal actual object PlatformSparseVectorKernels : SparseVectorKernels {
    actual override val name: String get() = "reference"

    /** The interface default, spelled out because an `expect object` member needs an `actual`. */
    actual override fun dot(x: SparseVector, y: DoubleArray): Double = super.dot(x, y)
}
