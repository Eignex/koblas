package com.eignex.koblas.dense

import com.eignex.koblas.StridedVector
import com.eignex.koblas.vendor.Blas

/**
 * Level 1 served by the vendor above a width, and by the portable kernels below it.
 *
 * The vendor implements all of Level 1 except [sum], which is not a BLAS routine, so on a target with no
 * vectorised Kotlin the library is the faster arm for anything but a short run: Kotlin/Native emits a
 * safepoint poll and a bounds check per element and does not vectorise these loops at all, while the binding
 * pins the caller's array and passes it in place with no copy in either direction.
 *
 * What it costs is a foreign call and two operand wrappers per invocation, which is a fixed price against
 * arithmetic that grows with the width. [CROSSOVER] is where the arithmetic starts paying for it, and below
 * that the portable kernels run instead, which is also what happens when the operation has no vendor entry
 * point at all.
 *
 * Not used on the JVM. There every operand is copied into native memory to reach the library, so the call
 * costs a pass over the data before it computes anything, and the Vector API kernels win at every width.
 */
internal class VendorVectorKernels(
    private val blas: Blas,
    private val portable: DenseVectorKernels = ScalarVectorKernels,
) : DenseVectorKernels {

    override val name: String get() = "${blas.vendor.vendorName.lowercase()}-level1"

    private fun vendorRuns(len: Int): Boolean = len >= CROSSOVER

    override fun implementationFor(operation: DenseOperation, length: Int, contiguous: Boolean): String? = when {
        // Not a BLAS routine, so there is no entry point to reach at any width.
        operation == DenseOperation.Sum -> portable.implementationFor(operation, length, contiguous)

        vendorRuns(length) -> name

        else -> portable.implementationFor(operation, length, contiguous)
    }

    private fun vector(v: DoubleArray, off: Int, len: Int, stride: Int) = StridedVector(v, off, len, stride)

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = if (vendorRuns(len)) {
        blas.dot(vector(a, aOff, len, aStride), vector(b, bOff, len, bStride))
    } else {
        portable.dot(a, aOff, b, bOff, len, aStride, bStride)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        if (vendorRuns(len)) blas.nrm2(vector(v, vOff, len, vStride)) else portable.nrm2(v, vOff, len, vStride)

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        if (vendorRuns(len)) blas.asum(vector(v, vOff, len, vStride)) else portable.asum(v, vOff, len, vStride)

    /**
     * The library's answer, which on a NaN is not the portable one.
     *
     * `idamax` does not specify what happens when the largest magnitude is not a number: the reference
     * compares strictly, so a NaN loses to a later finite entry, while oneMKL returns the NaN's index. This
     * routes like everything else rather than carving out an exception, so which of the two a caller sees is
     * the selected implementation's, the same way the Level 2 and 3 arithmetic is.
     */
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int =
        if (vendorRuns(len)) blas.iamax(vector(v, vOff, len, vStride)) else portable.iamax(v, vOff, len, vStride)

    /** Not a BLAS routine, so this is always the portable loop. */
    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = portable.sum(v, vOff, len, vStride)

    @Suppress("LongParameterList")
    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) {
        if (vendorRuns(len)) {
            blas.axpy(alpha, vector(x, xOff, len, xStride), vector(y, yOff, len, yStride))
        } else {
            portable.axpy(y, yOff, alpha, x, xOff, len, yStride, xStride)
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        if (vendorRuns(len)) {
            blas.scal(alpha, vector(v, vOff, len, vStride))
        } else {
            portable.scale(v, vOff, alpha, len, vStride)
        }
    }

    @Suppress("LongParameterList")
    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        if (vendorRuns(len)) {
            blas.swap(vector(a, aOff, len, aStride), vector(b, bOff, len, bStride))
        } else {
            portable.swap(a, aOff, b, bOff, len, aStride, bStride)
        }
    }

    /**
     * The library's, except over one run twice, which it would answer differently.
     *
     * [DenseVectorKernels.rot] promises that each pair is loaded before either result is stored, so equal runs
     * are safe. Reference `drot` stores `x` and then reads `x` again to form `y`, so handing it one pointer
     * twice leaves `(c + s) * v` where the portable and vectorised kernels leave `(c - s) * v`. That is the
     * caller's own contract rather than something the standard leaves open, so the portable kernel keeps it.
     */
    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        val oneRunTwice = x === y && xOff == yOff
        if (vendorRuns(len) && !oneRunTwice) {
            blas.rot(vector(x, xOff, len, 1), vector(y, yOff, len, 1), c, s)
        } else {
            portable.rot(x, xOff, y, yOff, len, c, s)
        }
    }

    internal companion object {
        /**
         * Width from which a foreign call beats the portable loop, to be replaced by a measured value.
         *
         * Provisional. The shape of the trade is not in doubt, a fixed per-call cost against arithmetic that
         * grows with the width, but where it crosses is a property of the host and the library and belongs in
         * a benchmark rather than in a guess. This sits deliberately high so that a short run, where the call
         * overhead dominates and the portable loop is certainly faster, is never sent across the boundary.
         */
        const val CROSSOVER: Int = 256
    }
}
