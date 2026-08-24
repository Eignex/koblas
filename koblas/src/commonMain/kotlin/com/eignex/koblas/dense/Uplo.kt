package com.eignex.koblas.dense

/** Triangle selector. [FULL] is a koblas extension to the lower/upper choice in BLAS and LAPACK. */
public enum class Uplo {
    /** Read or write the complete, exactly symmetric matrix. */
    FULL,

    /** Standard `dsyrk` with `uplo = L`: write and beta-scale the lower triangle only. */
    LOWER,

    /** Standard `dsyrk` with `uplo = U`: write and beta-scale the upper triangle only. */
    UPPER,
}

/** Accumulate `v` at `(i, j)` into whichever triangles [uplo] selects, the mirror too under [Uplo.FULL]. */
internal fun addUplo(cd: DoubleArray, n: Int, i: Int, j: Int, v: Double, uplo: Uplo) {
    if (uplo != Uplo.UPPER) {
        cd[i + j * n] += v
    } else if (i == j) {
        cd[i + i * n] += v // the diagonal belongs to both triangles
    }
    if (uplo != Uplo.LOWER && i != j) cd[j + i * n] += v
}

/** `v = beta * v` over the [len] entries from [off], honoring the `beta == 0` overwrite convention. */
internal fun applyBeta(k: F64Kernels, v: DoubleArray, off: Int, len: Int, beta: Double) {
    when {
        beta == 0.0 -> v.fill(0.0, off, off + len)
        beta != 1.0 -> k.scale(v, off, beta, len)
    }
}

/** `beta` scale of the region [uplo] selects, honoring the `beta == 0` overwrite convention. */
internal fun scaleUplo(k: F64Kernels, cd: DoubleArray, n: Int, beta: Double, uplo: Uplo) {
    if (uplo == Uplo.FULL) {
        applyBeta(k, cd, 0, cd.size, beta)
        return
    }
    if (beta == 1.0) return
    for (j in 0 until n) {
        val from = if (uplo == Uplo.LOWER) j + j * n else j * n
        val len = if (uplo == Uplo.LOWER) n - j else j + 1
        applyBeta(k, cd, from, len, beta)
    }
}
