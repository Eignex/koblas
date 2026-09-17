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
 * arithmetic that grows with the width. [crossover] is where the arithmetic starts paying for it, and below
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

    private fun vendorRuns(operation: DenseOperation, len: Int): Boolean = len >= crossover(operation)

    override fun implementationFor(operation: DenseOperation, length: Int, contiguous: Boolean): String? = when {
        // Not a BLAS routine, so there is no entry point to reach at any width.
        operation == DenseOperation.Sum -> portable.implementationFor(operation, length, contiguous)

        vendorRuns(operation, length) -> name

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
    ): Double = if (vendorRuns(DenseOperation.Dot, len)) {
        blas.dot(vector(a, aOff, len, aStride), vector(b, bOff, len, bStride))
    } else {
        portable.dot(a, aOff, b, bOff, len, aStride, bStride)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        if (vendorRuns(DenseOperation.Nrm2, len)) {
            blas.nrm2(vector(v, vOff, len, vStride))
        } else {
            portable.nrm2(v, vOff, len, vStride)
        }

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        if (vendorRuns(DenseOperation.Asum, len)) {
            blas.asum(vector(v, vOff, len, vStride))
        } else {
            portable.asum(v, vOff, len, vStride)
        }

    /**
     * The library's answer, which on a NaN is not the portable one.
     *
     * `idamax` does not specify what happens when the largest magnitude is not a number: the reference
     * compares strictly, so a NaN loses to a later finite entry, while oneMKL returns the NaN's index. This
     * routes like everything else rather than carving out an exception, so which of the two a caller sees is
     * the selected implementation's, the same way the Level 2 and 3 arithmetic is.
     */
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int =
        if (vendorRuns(DenseOperation.Iamax, len)) {
            blas.iamax(vector(v, vOff, len, vStride))
        } else {
            portable.iamax(v, vOff, len, vStride)
        }

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
        if (vendorRuns(DenseOperation.Axpy, len)) {
            blas.axpy(alpha, vector(x, xOff, len, xStride), vector(y, yOff, len, yStride))
        } else {
            portable.axpy(y, yOff, alpha, x, xOff, len, yStride, xStride)
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        if (vendorRuns(DenseOperation.Scale, len)) {
            blas.scal(alpha, vector(v, vOff, len, vStride))
        } else {
            portable.scale(v, vOff, alpha, len, vStride)
        }
    }

    @Suppress("LongParameterList")
    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        if (vendorRuns(DenseOperation.Swap, len)) {
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
        if (vendorRuns(DenseOperation.Rot, len) && !oneRunTwice) {
            blas.rot(vector(x, xOff, len, 1), vector(y, yOff, len, 1), c, s)
        } else {
            portable.rot(x, xOff, y, yOff, len, c, s)
        }
    }

    internal companion object {
        /**
         * The width from which the foreign call beats the portable loop for [operation].
         *
         * Three values rather than one, because the measured break-evens do not overlap: they run from 205 to
         * 652, and the operations at the ends differ from the middle for reasons that are not noise. What
         * decides the width is how much work each side does per element, so operations that share an answer
         * share it for a reason and not by rounding.
         *
         * Measured on 12th Gen Intel Core i9-12900H, P-cores 2/4/6/8 pinned, against oneMKL 2026.1 held to one
         * compute thread, Kotlin/Native 2.4.10 linuxX64. The capture, its settings, the per-width ratios and
         * the resolved library file are in `koblas-bench/reports/level1-crossover/`, over widths 8 to 262144.
         * A coarse pass at a quarter of those settings put every break-even within 8% of these.
         *
         * No other host has been measured. Another CPU or another library keeps these numbers only until
         * someone runs that sweep there, and nothing here is inferred from a backend that was not run.
         */
        fun crossover(operation: DenseOperation): Int = when (operation) {
            DenseOperation.Iamax, DenseOperation.Rot -> LOOP_HEAVY
            DenseOperation.Nrm2 -> ROBUST_NORM
            else -> STREAMING
        }

        /**
         * One multiply or add per element on each side: `dot` 354, `asum` 365, `axpy` 373, `scal` 331, `swap`
         * 403.
         *
         * Rounded up to the swept width from which every one of them is ahead rather than down to the earliest,
         * because the two directions do not cost the same. Sending a call across too early makes it slower than
         * the loop it replaced, which is a regression a caller sees; holding it back only forgoes a win. At 384
         * `swap` is still 0.97, which is the one operation this value is marginally early for, and the same
         * value is 1.02 to 1.12 for the other four.
         */
        const val STREAMING: Int = 384

        /**
         * Several operations per element in the portable loop, so the call is repaid sooner: `iamax` 205,
         * `rot` 236.
         *
         * `rot` computes two results from each pair and `iamax` carries a running index beside the magnitude,
         * where the streaming kernels do one arithmetic step and move on. The portable side is what is slow
         * here, not the library that is fast: by 1024 these reach 3.6 and 2.5 against the loop.
         */
        const val LOOP_HEAVY: Int = 256

        /**
         * `nrm2` alone, at 652, because here it is the library doing the extra work.
         *
         * `dnrm2` rescales for overflow safety as it goes; the portable kernel tries the plain sum of squares
         * first and only falls back to a rescaling loop when a value leaves the normal range. So the vendor
         * carries a cost the portable loop usually does not pay, and it is repaid latest and least: 2.06 at
         * the widest measured width, against 4 to 9 for the plain reductions. At [STREAMING] the library would
         * still be 26% slower than the loop it replaced.
         */
        const val ROBUST_NORM: Int = 768
    }
}
