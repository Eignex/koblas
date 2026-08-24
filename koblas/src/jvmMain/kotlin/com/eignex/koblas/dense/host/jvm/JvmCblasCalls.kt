package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.CblasCalls

/**
 * [CblasCalls] over `java.lang.foreign`. Each array goes across as a segment over the on-heap array itself,
 * which `Linker.Option.critical` lets the call read and write in place rather than copying.
 */
@Suppress("LongParameterList") // the CBLAS signatures
internal class JvmCblasCalls(private val calls: HostBlasCalls) : CblasCalls {
    private fun seg(values: DoubleArray) = java.lang.foreign.MemorySegment.ofArray(values)
    override fun dscal(n: Int, alpha: Double, x: DoubleArray, incx: Int) {
        calls.dscal.invokeExact(n, alpha, seg(x), incx) as Unit
    }

    override fun daxpy(n: Int, alpha: Double, x: DoubleArray, incx: Int, y: DoubleArray, incy: Int) {
        calls.daxpy.invokeExact(n, alpha, seg(x), incx, seg(y), incy) as Unit
    }

    override fun dgemv(
        order: Int,
        trans: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        x: DoubleArray,
        incx: Int,
        beta: Double,
        y: DoubleArray,
        incy: Int,
    ) {
        calls.dgemv.invokeExact(
            order, trans, m, n, alpha, seg(a), lda, seg(x), incx, beta, seg(y), incy,
        ) as Unit
    }

    override fun dger(
        order: Int,
        m: Int,
        n: Int,
        alpha: Double,
        x: DoubleArray,
        incx: Int,
        y: DoubleArray,
        incy: Int,
        a: DoubleArray,
        lda: Int,
    ) {
        calls.dger.invokeExact(order, m, n, alpha, seg(x), incx, seg(y), incy, seg(a), lda) as Unit
    }

    override fun dsymv(
        order: Int,
        uplo: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        x: DoubleArray,
        incx: Int,
        beta: Double,
        y: DoubleArray,
        incy: Int,
    ) {
        calls.dsymv.invokeExact(order, uplo, n, alpha, seg(a), lda, seg(x), incx, beta, seg(y), incy) as Unit
    }

    override fun dtrsv(
        order: Int,
        uplo: Int,
        trans: Int,
        diag: Int,
        n: Int,
        a: DoubleArray,
        lda: Int,
        x: DoubleArray,
        incx: Int,
    ) {
        calls.dtrsv.invokeExact(order, uplo, trans, diag, n, seg(a), lda, seg(x), incx) as Unit
    }

    override fun dtrmv(
        order: Int,
        uplo: Int,
        trans: Int,
        diag: Int,
        n: Int,
        a: DoubleArray,
        lda: Int,
        x: DoubleArray,
        incx: Int,
    ) {
        calls.dtrmv.invokeExact(order, uplo, trans, diag, n, seg(a), lda, seg(x), incx) as Unit
    }

    override fun dgemm(
        order: Int,
        transA: Int,
        transB: Int,
        m: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    ) {
        calls.dgemm.invokeExact(
            order, transA, transB, m, n, k, alpha, seg(a), lda, seg(b), ldb, beta, seg(c), ldc,
        ) as Unit
    }

    override fun dsyrk(
        order: Int,
        uplo: Int,
        trans: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    ) {
        calls.dsyrk.invokeExact(order, uplo, trans, n, k, alpha, seg(a), lda, beta, seg(c), ldc) as Unit
    }

    override fun dsymm(
        order: Int,
        side: Int,
        uplo: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    ) {
        calls.dsymm.invokeExact(
            order, side, uplo, m, n, alpha, seg(a), lda, seg(b), ldb, beta, seg(c), ldc,
        ) as Unit
    }

    override fun dtrsm(
        order: Int,
        side: Int,
        uplo: Int,
        trans: Int,
        diag: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
    ) {
        calls.dtrsm.invokeExact(order, side, uplo, trans, diag, m, n, alpha, seg(a), lda, seg(b), ldb) as Unit
    }

    override fun dtrmm(
        order: Int,
        side: Int,
        uplo: Int,
        trans: Int,
        diag: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
    ) {
        calls.dtrmm.invokeExact(order, side, uplo, trans, diag, m, n, alpha, seg(a), lda, seg(b), ldb) as Unit
    }
}
