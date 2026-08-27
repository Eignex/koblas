package com.eignex.koblas.dense

/** Accumulate [value] at `(i, j)` in the selected triangle, where `i >= j`. */
internal fun addTriangle(cd: DoubleArray, n: Int, i: Int, j: Int, value: Double, lower: Boolean) {
    if (lower) cd[i + j * n] += value else cd[j + i * n] += value
}

/** `v = beta * v` over the [len] entries from [off], honoring the `beta == 0` overwrite convention. */
internal fun applyBeta(k: F64Kernels, v: DoubleArray, off: Int, len: Int, beta: Double) {
    when {
        beta == 0.0 -> v.fill(0.0, off, off + len)
        beta != 1.0 -> k.scale(v, off, beta, len)
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
