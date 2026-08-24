package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.LapackeCalls

/** [LapackeCalls] over `java.lang.foreign`, alongside [JvmCblasCalls]. */
@Suppress("LongParameterList") // the LAPACKE signatures
internal class JvmLapackeCalls(private val calls: HostBlasCalls) : LapackeCalls {
    private fun seg(values: DoubleArray) = java.lang.foreign.MemorySegment.ofArray(values)
    private fun seg(values: IntArray) = java.lang.foreign.MemorySegment.ofArray(values)
    override fun dgetrf(order: Int, m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        calls.dgetrf.invokeExact(order, m, n, seg(a), lda, seg(ipiv)) as Int

    override fun dgecon(
        order: Int,
        norm: Byte,
        n: Int,
        a: DoubleArray,
        lda: Int,
        anorm: Double,
        rcond: DoubleArray,
    ): Int = calls.dgecon.invokeExact(order, norm, n, seg(a), lda, anorm, seg(rcond)) as Int

    override fun dgeqrf(order: Int, m: Int, n: Int, a: DoubleArray, lda: Int, tau: DoubleArray): Int =
        calls.dgeqrf.invokeExact(order, m, n, seg(a), lda, seg(tau)) as Int

    override fun dormqr(
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
    ): Int = calls.dormqr.invokeExact(
        order, side, trans, m, n, k, seg(a), lda, seg(tau), seg(c), ldc,
    ) as Int

    override fun dpotrf(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int): Int =
        calls.dpotrf.invokeExact(order, uplo, n, seg(a), lda) as Int

    override fun dpotri(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int): Int =
        calls.dpotri.invokeExact(order, uplo, n, seg(a), lda) as Int

    override fun dsytrf(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        calls.dsytrf.invokeExact(order, uplo, n, seg(a), lda, seg(ipiv)) as Int

    override fun dsytrs(
        order: Int,
        uplo: Byte,
        n: Int,
        nrhs: Int,
        a: DoubleArray,
        lda: Int,
        ipiv: IntArray,
        b: DoubleArray,
        ldb: Int,
    ): Int = calls.dsytrs.invokeExact(order, uplo, n, nrhs, seg(a), lda, seg(ipiv), seg(b), ldb) as Int
}
