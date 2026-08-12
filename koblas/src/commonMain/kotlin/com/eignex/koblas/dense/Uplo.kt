package com.eignex.koblas.dense

/** Output-triangle selector for [LinearAlgebra.syrk]. [FULL] is the default and a koblas extension. */
public enum class Uplo {
    /** Write the complete, exactly symmetric result; beta scales all of `C`. */
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

/** `beta` scale of the region [uplo] selects, honoring the `beta == 0` overwrite convention. */
internal fun scaleUplo(k: VectorKernels, cd: DoubleArray, n: Int, beta: Double, uplo: Uplo) {
    if (uplo == Uplo.FULL) {
        if (beta == 0.0) {
            cd.fill(0.0)
        } else if (beta != 1.0) {
            k.scale(cd, 0, beta, cd.size)
        }
        return
    }
    if (beta == 1.0) return
    for (j in 0 until n) {
        val from = if (uplo == Uplo.LOWER) j + j * n else j * n
        val len = if (uplo == Uplo.LOWER) n - j else j + 1
        if (beta == 0.0) cd.fill(0.0, from, from + len) else k.scale(cd, from, beta, len)
    }
}
