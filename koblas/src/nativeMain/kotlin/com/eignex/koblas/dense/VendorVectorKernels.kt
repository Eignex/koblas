package com.eignex.koblas.dense

import com.eignex.koblas.vendor.NativeVendorBlas

/**
 * Level 1 served by the vendor above a width, and by the portable kernels below it.
 *
 * The vendor implements all of Level 1 except [sum], which is not a BLAS routine, so on a target with no
 * vectorised Kotlin the library is the faster arm for anything but a short run. What the portable loop lacks
 * is vector arithmetic: Kotlin/Native has no Vector API and LLVM will not reorder a floating-point reduction
 * on its own, so it runs about half a nanosecond per element where the same dot in C runs at a tenth of that.
 * Reading the same array through a pinned pointer instead, which has no bounds check, measured five times
 * slower still, so the checks are not what to blame.
 *
 * What the call costs is a foreign call and a pin per operand, about ten nanoseconds each, which is a fixed
 * price against arithmetic that grows with the width. [crossover] is where the arithmetic starts paying for
 * it, and below that the portable kernels run instead, which is also what happens when the operation has no
 * vendor entry point at all.
 *
 * Not used on the JVM. There every operand is copied into native memory to reach the library, so the call
 * costs a pass over the data before it computes anything, and the Vector API kernels win at every width.
 */
internal class VendorVectorKernels(
    private val blas: NativeVendorBlas,
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

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = if (vendorRuns(DenseOperation.Dot, len)) {
        blas.rawDot(a, aOff, aStride, b, bOff, bStride, len)
    } else {
        portable.dot(a, aOff, b, bOff, len, aStride, bStride)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        if (vendorRuns(DenseOperation.Nrm2, len)) {
            blas.rawNrm2(v, vOff, vStride, len)
        } else {
            portable.nrm2(v, vOff, len, vStride)
        }

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        if (vendorRuns(DenseOperation.Asum, len)) {
            blas.rawAsum(v, vOff, vStride, len)
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
            blas.rawIamax(v, vOff, vStride, len)
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
            blas.rawAxpy(alpha, x, xOff, xStride, y, yOff, yStride, len)
        } else {
            portable.axpy(y, yOff, alpha, x, xOff, len, yStride, xStride)
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        if (vendorRuns(DenseOperation.Scale, len)) {
            blas.rawScal(alpha, v, vOff, vStride, len)
        } else {
            portable.scale(v, vOff, alpha, len, vStride)
        }
    }

    @Suppress("LongParameterList")
    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        if (vendorRuns(DenseOperation.Swap, len)) {
            blas.rawSwap(a, aOff, aStride, b, bOff, bStride, len)
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
            blas.rawRot(x, xOff, 1, y, yOff, 1, len, c, s)
        } else {
            portable.rot(x, xOff, y, yOff, len, c, s)
        }
    }

    internal companion object {
        /**
         * The width from which the foreign call beats the portable loop for [operation].
         *
         * Two values, because the eight measured break-evens fall into two groups with a gap between them:
         * `axpy` 40, `scal` 42, `asum` 45, `rot` 46, `dot` 57 and `swap` 67, then `nrm2` 82 and `iamax` 100.
         * The six share a shape. Each vendor call costs a fixed 33 to 60 nanoseconds whatever the width, and
         * the portable loop costs about half a nanosecond per element, so they meet where the loop's work
         * grows to the size of the call. The two stragglers are late for reasons of their own, given below.
         *
         * Measured on 12th Gen Intel Core i9-12900H, P-cores 2/4/6/8 pinned, against oneMKL 2026.1 held to one
         * compute thread, Kotlin/Native 2.4.10 linuxX64, over widths 1 to 262144. Reproduced with
         * `koblas-bench/capture-report.sh --libraries onemkl --suite sweep --operation <name> --samples 5
         * --warmups 5 --target-ms 200 --forks 2`, once per operation, comparing the `native` and `onemkl`
         * targets. Pinning matters on this part: its E-cores run a gigahertz slower and an unpinned run wanders
         * between the two.
         *
         * A crossover is a property of the call as much as of the arithmetic, so these hold only while a call
         * costs what it costs here: 33 nanoseconds for one pinned operand and 42 for two. Anything that moves
         * that figure calls for the sweep again rather than an adjustment to these.
         *
         * No other host has been measured. Another CPU or another library keeps these numbers only until
         * someone runs that sweep there, and nothing here is inferred from a backend that was not run.
         */
        fun crossover(operation: DenseOperation): Int = when (operation) {
            DenseOperation.Nrm2, DenseOperation.Iamax -> LATE
            else -> ORDINARY
        }

        /**
         * `axpy`, `scal`, `asum`, `rot`, `dot` and `swap`, measuring 40 to 67.
         *
         * Rounded to where the latest of them arrives rather than the earliest, because the two directions do
         * not cost the same: sending a call across too early makes it slower than the loop it replaced, which
         * a caller sees, while holding it back only forgoes a win. At 64 the six run 0.99 to 1.30 against the
         * portable loop, so none is sent early and the most any of them gives up is a few per cent between its
         * own break-even and this width.
         */
        const val ORDINARY: Int = 64

        /**
         * `nrm2` at 82 and `iamax` at 100, which arrive late for unrelated reasons.
         *
         * `dnrm2` rescales for overflow safety as it goes, so unlike the others its cost grows with the width
         * from the start: 37.8 nanoseconds at 8 entries and 73.6 at 128. The portable kernel tries the plain
         * sum of squares first and only falls back to a rescaling loop when a value leaves the normal range,
         * so it is doing less work, and the library's advantage is both later and smaller, reaching 2.06 where
         * the plain reductions reach 4 to 9.
         *
         * `idamax` is the opposite: its entry costs 60 to 120 nanoseconds before any width matters, measured
         * against the bare C call and so not this binding's doing, where `dasum` over the same operand costs 9.
         * One value covers both, since the difference between 82 and 100 is worth less than a third constant.
         */
        const val LATE: Int = 128
    }
}
