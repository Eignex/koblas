package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.internal.numeric.scalarAxpy
import com.eignex.koblas.internal.numeric.scalarDot
import com.eignex.koblas.internal.numeric.scalarIamax
import com.eignex.koblas.internal.numeric.scalarRot
import com.eignex.koblas.internal.numeric.scalarScale
import com.eignex.koblas.internal.numeric.scalarSum
import com.eignex.koblas.internal.numeric.scalarSwap

/**
 * Contiguous dense Level 1 operations that are independently useful outside matrix algorithms.
 * Default methods use portable Kotlin arithmetic. Implementations must read and write only the array windows
 * selected by their offsets and lengths.
 * Callers must supply non-negative lengths and valid windows; implementations need not validate them.
 * A zero length is legal for every operation that accepts one and produces the documented empty result
 * without reading either array.
 *
 * Results agree to within rounding rather than bit for bit: compiled leaves may fuse multiplication and
 * addition or reduce vector lanes as a tree. Shared algorithms must not depend on the final rounding order.
 */
public interface DenseVectorKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /** Returns `sum(a[aOff + i] * b[bOff + i])` over [len] entries, or zero for an empty run. */
    public fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Double = scalarDot(a, aOff, aStride, b, bOff, bStride, len)

    /**
     * Adds `alpha * x[xOff + i]` into `y[yOff + i]` over [len] entries. A zero [alpha] returns without
     * evaluating the products, as required by standalone BLAS AXPY semantics.
     */
    public fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int = 1,
        xStride: Int = 1,
    ): Unit = scalarAxpy(y, yOff, yStride, alpha, x, xOff, xStride, len)

    /** Scales [len] entries in [v] by [alpha]. */
    public fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int = 1): Unit =
        scalarScale(v, vOff, vStride, alpha, len)

    /**
     * Returns the rescaled Euclidean norm over [len] entries, or zero for an empty run. Implementations
     * must avoid the avoidable overflow and underflow of a plain square-sum followed by a square root.
     */
    public fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double =
        euclideanNorm(v, vOff, vStride, len)

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    public fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = absoluteSum(v, vOff, vStride, len)

    /**
     * Zero-based index within the selected run of a maximum absolute value, or `-1` when empty. A nonempty
     * run of zeros returns `0`.
     *
     * Which index is returned when the maximum is not unique, and what happens when the largest magnitude is
     * a NaN, is this implementation's. `idamax` specifies neither: the portable kernels compare strictly, so
     * ties go to the first and a NaN loses to a later finite entry, while a vendor may report the NaN's index
     * and may resolve a tie the other way for a negatively spaced run.
     */
    public fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Int = scalarIamax(v, vOff, vStride, len)

    /**
     * Applies the plane rotation `(c, s, -s, c)` to two contiguous runs. Each pair is loaded before either
     * result is stored. Equal runs are therefore safe, but callers must snapshot any other overlap whose
     * stores could precede a later input load.
     */
    @Suppress("LongParameterList")
    public fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double): Unit =
        scalarRot(x, xOff, y, yOff, len, c, s)

    /** Exchanges [len] entries of the two runs. */
    public fun swap(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = scalarSwap(a, aOff, aStride, b, bOff, bStride, len)

    /**
     * Returns the plain sum over [len] entries, or zero for an empty run. This is not compensated summation;
     * compiled leaves may reduce vector lanes as a tree.
     */
    public fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = scalarSum(v, vOff, vStride, len)
}
