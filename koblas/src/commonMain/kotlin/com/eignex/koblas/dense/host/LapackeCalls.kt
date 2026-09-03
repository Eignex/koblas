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

    fun dgeqp3(order: Int, m: Int, n: Int, a: DoubleArray, lda: Int, jpvt: IntArray, tau: DoubleArray): Int

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

    /**
     * The blocked QR of the triangular-pentagonal `[a; b]`. [a] is the `n` by `n` upper triangle and comes
     * back as `R`; [b] is the `m` by `n` block below it, rectangular at `l = 0`, and comes back holding the
     * reflectors; [t] is the `nb` by `n` block of triangular factors.
     */
    fun dtpqrt(
        order: Int,
        m: Int,
        n: Int,
        l: Int,
        nb: Int,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        t: DoubleArray,
        ldt: Int,
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
