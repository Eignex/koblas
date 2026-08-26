package com.eignex.koblas.dense.host

/**
 * The LAPACKE entry points koblas binds, in plain arrays and ints, alongside [CblasCalls]. Each returns
 * LAPACK's `info`: negative for an illegal argument, positive for a numerical failure whose meaning is the
 * routine's own.
 *
 * An implementation must not copy its arrays, since the routines write their results through them.
 */
@Suppress("LongParameterList")
internal interface LapackeCalls {
    fun dgetrf(order: Int, m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int

    fun dgecon(order: Int, norm: Byte, n: Int, a: DoubleArray, lda: Int, anorm: Double, rcond: DoubleArray): Int

    fun dgeqrf(order: Int, m: Int, n: Int, a: DoubleArray, lda: Int, tau: DoubleArray): Int

    fun dormqr(
        order: Int,
        side: Byte,
        trans: Byte,
        m: Int,
        n: Int,
        k: Int,
        a: DoubleArray,
        lda: Int,
        tau: DoubleArray,
        c: DoubleArray,
        ldc: Int,
    ): Int

    fun dgetri(order: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int

    fun dtrtri(order: Int, uplo: Byte, diag: Byte, n: Int, a: DoubleArray, lda: Int): Int

    fun dpotrf(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int): Int

    fun dpotri(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int): Int

    fun dsytrf(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int

    fun dsytrs(
        order: Int,
        uplo: Byte,
        n: Int,
        nrhs: Int,
        a: DoubleArray,
        lda: Int,
        ipiv: IntArray,
        b: DoubleArray,
        ldb: Int,
    ): Int
}
