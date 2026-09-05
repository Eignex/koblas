package com.eignex.koblas.dense.host

/**
 * The level-1 CBLAS entry points koblas binds, in plain arrays, element offsets and ints. The counterpart of
 * [CblasCalls] one level down: the two host bindings differ only in how they hand an array to the library, so
 * this is the seam between that mechanism and the routines built on it.
 *
 * Every call names an element offset into the caller's array, since a kernel runs over a window of a longer
 * buffer rather than over the whole of it. An implementation must not copy, since [F64KernelsAdapter] relies
 * on the library writing through to the caller's arrays.
 */
@Suppress("LongParameterList", "TooManyFunctions") // the CBLAS signatures, one method each
internal interface CblasKernelCalls {
    fun ddot(n: Int, x: DoubleArray, xOffset: Int, incx: Int, y: DoubleArray, yOffset: Int, incy: Int): Double

    fun daxpy(n: Int, alpha: Double, x: DoubleArray, xOffset: Int, incx: Int, y: DoubleArray, yOffset: Int, incy: Int)

    fun dswap(n: Int, x: DoubleArray, xOffset: Int, incx: Int, y: DoubleArray, yOffset: Int, incy: Int)

    fun dscal(n: Int, alpha: Double, x: DoubleArray, xOffset: Int, incx: Int)

    fun dnrm2(n: Int, x: DoubleArray, xOffset: Int, incx: Int): Double

    fun dasum(n: Int, x: DoubleArray, xOffset: Int, incx: Int): Double

    /** [parameters] is the five-element `DPARAM` block, read from its start. */
    fun drotm(
        n: Int,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
        parameters: DoubleArray,
    )

    fun drot(
        n: Int,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
        c: Double,
        s: Double,
    )
}
