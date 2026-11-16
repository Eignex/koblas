@file:Suppress("LongParameterList") // Primitive arithmetic windows avoid per-call descriptors.

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*

/** Dense vector arithmetic; capabilities select which optional operations may be called. */
internal interface VectorKernels : KernelProvider {

    /** Returns `sum(a[aOff + i] * b[bOff + i])` over [len] entries, or zero for an empty run. */
    fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Double = missingKernel(name)

    /**
     * Adds `alpha * x[xOff + i]` into `y[yOff + i]` over [len] entries. A zero [alpha] returns without
     * evaluating the products, as required by standalone BLAS AXPY semantics.
     */
    fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int = 1,
        xStride: Int = 1,
    ): Unit = missingKernel(name)

    /** Scales [len] entries in [v] by [alpha]. */
    fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int = 1): Unit = missingKernel(name)

    /**
     * Returns the rescaled Euclidean norm over [len] entries, or zero for an empty run. Implementations
     * must avoid the avoidable overflow and underflow of a plain square-sum followed by a square root.
     */
    fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = missingKernel(name)

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = missingKernel(name)

    /**
     * Zero-based index within the selected run of a maximum absolute value, or `-1` when empty. A nonempty
     * run of zeros returns `0`.
     *
     * Which index is returned when the maximum is not unique, and what happens when the largest magnitude is
     * a NaN, is this implementation's. `idamax` specifies neither: the portable kernels compare strictly, so
     * ties go to the first and a NaN loses to a later finite entry, while a vendor may report the NaN's index
     * and may resolve a tie the other way for a negatively spaced run.
     */
    fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Int = missingKernel(name)

    /**
     * Applies the plane rotation `(c, s, -s, c)` to two primitive vector runs. Each pair is loaded before either
     * result is stored. Equal runs are therefore safe, but callers must snapshot any other overlap whose
     * stores could precede a later input load.
     */
    @Suppress("LongParameterList")
    fun rot(
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        len: Int,
        c: Double,
        s: Double,
        xStride: Int = 1,
        yStride: Int = 1,
    ): Unit = missingKernel(name)

    /** Exchanges [len] entries of the two runs. */
    fun swap(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = missingKernel(name)

    /** Copies a primitive vector window. Callers stage destructive overlap before dispatch. */
    fun copy(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = missingKernel(name)

    /**
     * Returns the plain sum over [len] entries, or zero for an empty run. This is not compensated summation;
     * compiled leaves may reduce vector lanes as a tree.
     */
    fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = missingKernel(name)
}

private fun missingKernel(name: String): Nothing =
    throw UnsupportedOperationException("This operation is not supplied by $name")
