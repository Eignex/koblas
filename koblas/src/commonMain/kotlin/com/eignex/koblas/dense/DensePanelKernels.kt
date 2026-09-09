package com.eignex.koblas.dense

/**
 * Arithmetic over contiguous matrix columns and panels. Callers must supply non-negative lengths and valid
 * array windows; implementations need not validate them. A zero length is legal and must not read an array.
 * Unlike standalone BLAS AXPY, update operations in this family evaluate zero multipliers so matrix arithmetic
 * preserves products such as `0 * infinity`.
 */
public interface DensePanelKernels {
    /**
     * Writes four dot products against one shared right operand into [out]. For `r` in `0..3`,
     * `out[outOff + r]` becomes the dot product of the [len] entries beginning at `aOff + r * stride` and
     * `bOff`. The four outputs are written only after their input reductions are complete.
     */
    @Suppress("LongParameterList")
    public fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    )

    /**
     * Adds four scaled, equally spaced matrix columns into [y]. For every `i` in the run, the operation is
     * `y[yOff + i] += c0*a[aOff + i] + c1*a[aOff + stride + i] + c2*a[aOff + 2*stride + i] +
     * c3*a[aOff + 3*stride + i]`, evaluated in coefficient order. Zero coefficients are evaluated.
     */
    @Suppress("LongParameterList")
    public fun axpy4(
        y: DoubleArray,
        yOff: Int,
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        c0: Double,
        c1: Double,
        c2: Double,
        c3: Double,
        len: Int,
    )

    /**
     * Returns the dot product of the [a] and [x] runs while adding `alpha * a` into [y] in the same pass.
     * Each [a] and [x] element is loaded before the corresponding destination is stored, so an input may equal
     * the destination run. Callers must otherwise avoid overlap whose stores precede later input loads. Zero
     * [alpha] is evaluated.
     */
    @Suppress("LongParameterList")
    public fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double

    /**
     * Adds `alpha * x[xOff + i]` into `y[yOff + i]` over [len] entries and always evaluates the product,
     * including when [alpha] is zero. Equal input and destination runs are safe; callers must otherwise avoid
     * overlap whose stores precede later input loads.
     */
    @Suppress("LongParameterList")
    public fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)
}
