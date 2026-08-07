package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector

/**
 * The sparse level-1 kernels compiled in for this target, the counterpart of the dense
 * `PlatformVectorKernels`.
 *
 * Every target but the JVM inherits the interface defaults, which are the portable loops. The JVM overrides
 * `dot` against a dense operand with a gather: `y[indices[k]]` for a whole vector register at once, then one
 * fused multiply-add. Measured on an i9-12900H (AVX2, four lanes) under JDK 25, that is 1.17x at nnz 128,
 * 1.51x at 512 and 1.61x at 2048, and a 15 to 25 per cent *loss* at nnz 8 to 32 where the prologue is not
 * covered — hence a threshold rather than an unconditional override.
 *
 * The three other indexed routines are deliberately left portable. `usaxpy` and the scatter have to write at
 * `indices[k]`, and AVX2 has no double scatter, so the Vector API emulates it lane by lane: measured 4x to
 * 15x *slower* at every size tried. A sparse-against-sparse `dot` merges two ascending index lists with
 * data-dependent branching, which has no vector form at all.
 *
 * An `expect object` rather than a registered backend, matching the dense tier: these are koblas's own
 * compiled-in kernels on every target, not something a host library provides, so they are always present and
 * a registered [SparseVectorKernels] outranks them through the ordinary seam.
 */
internal expect object PlatformSparseVectorKernels : SparseVectorKernels {
    override val name: String

    /** `xᵀ·y` against a dense operand — the one routine a target may accelerate. */
    override fun dot(x: SparseVector, y: DoubleArray): Double
}
