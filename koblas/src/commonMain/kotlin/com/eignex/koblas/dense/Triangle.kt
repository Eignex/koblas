package com.eignex.koblas.dense

/** `v = beta * v` over the [len] entries from [off], honoring the `beta == 0` overwrite convention. */
internal fun applyBeta(k: F64Kernels, v: DoubleArray, off: Int, len: Int, beta: Double) {
    when {
        beta == 0.0 -> v.fill(0.0, off, off + len)
        beta != 1.0 -> k.scale(v, off, beta, len)
    }
}

/**
 * `y += alpha * x` for a parent BLAS routine that does not have DAXPY's `alpha == 0` quick return.
 * Nonzero multipliers retain the selected Level 1 kernel; an exact zero forms the products explicitly so
 * `0 * infinity` remains a NaN as it is in the reference parent routine.
 */
internal fun axpyArithmetic(
    k: F64Kernels,
    y: DoubleArray,
    yOff: Int,
    alpha: Double,
    x: DoubleArray,
    xOff: Int,
    len: Int,
) {
    if (alpha != 0.0) {
        k.axpy(y, yOff, alpha, x, xOff, len)
    } else if (k is F64ArithmeticKernels) {
        k.axpyArithmetic(y, yOff, alpha, x, xOff, len)
    } else {
        for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
    }
}

/** Scale the selected triangle by [beta], honoring the `beta == 0` overwrite convention. */
internal fun scaleTriangle(k: F64Kernels, cd: DoubleArray, n: Int, beta: Double, lower: Boolean) {
    if (beta == 1.0) return
    for (j in 0 until n) {
        val from = if (lower) j + j * n else j * n
        val len = if (lower) n - j else j + 1
        applyBeta(k, cd, from, len, beta)
    }
}
