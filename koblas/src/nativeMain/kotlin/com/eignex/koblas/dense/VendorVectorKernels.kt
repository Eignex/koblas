package com.eignex.koblas.dense

import com.eignex.koblas.vendor.NativeVendorBlas

/**
 * Vendor Level 1 above [crossover], portable kernels below it or without a vendor entry point.
 *
 * Kotlin/Native reductions lack explicit vector arithmetic. Vendor calls amortize their fixed
 * foreign-call and operand-pinning costs over longer runs. JVM bindings copy operands into native
 * memory instead, so they use a separate engine composition.
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

        // A spaced rotation has no kernel here and is the portable loop. A contiguous one keeps a run rotated
        // against itself portable at every width, and a route carries no operand identity, so this call's
        // facts do not settle which of the two it is.
        operation == DenseOperation.Rot ->
            if (contiguous && vendorRuns(operation, length)) {
                null
            } else {
                portable.implementationFor(operation, length, contiguous)
            }

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
     * `idamax` leaves NaN handling unspecified. The portable strict comparison and a vendor's result
     * can differ; [iamax] uses the selected implementation's answer.
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
     * Equal runs use the portable kernel to honor [DenseVectorKernels.rot]'s load-before-store
     * contract. Reference `drot` stores `x` before reading it for `y`, which changes the result when
     * both pointers address the same run.
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
         * Measured foreign-call crossover for [operation]. Ordinary operations broke even at widths
         * 40 to 67; [DenseOperation.Nrm2] and [DenseOperation.Iamax] at 82 and 100.
         *
         * Measured with the koblas-bench sweep suite on an i9-12900H with pinned P-cores, single-threaded
         * oneMKL 2026.1 and Kotlin/Native 2.4.10 linuxX64. Changes to the CPU, library or binding overhead
         * require a new sweep; these thresholds are not evidence for unmeasured hosts.
         */
        fun crossover(operation: DenseOperation): Int = when (operation) {
            DenseOperation.Nrm2, DenseOperation.Iamax -> LATE
            else -> ORDINARY
        }

        /**
         * Conservative threshold for ordinary operations: calling the vendor too early can lose time,
         * while delaying the call only forgoes a small win near the crossover.
         */
        const val ORDINARY: Int = 64

        /**
         * [DenseOperation.Nrm2] pays for rescaling while the portable fast path tries a plain square sum.
         * [DenseOperation.Iamax] has higher vendor entry overhead. Their nearby crossovers share one threshold.
         */
        const val LATE: Int = 128
    }
}
