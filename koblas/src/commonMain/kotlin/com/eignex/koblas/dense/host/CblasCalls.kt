package com.eignex.koblas.dense.host

/**
 * The CBLAS entry points koblas binds, in plain arrays and ints. The two host bindings differ only in how
 * they hand an array to the library, so this is the seam between that mechanism and the routines built on
 * it: the JVM implements it over `java.lang.foreign`, Kotlin/Native over `usePinned`.
 *
 * Contiguous calls pass an array from its start. Strided view overloads additionally advance the native
 * pointer by an element offset; leading dimensions and increments remain separate arguments, as in CBLAS
 * itself. An implementation must not copy, since [F64BlasAdapter] relies on the library writing through to
 * the caller's arrays.
 */
@Suppress("LongParameterList", "TooManyFunctions") // the CBLAS signatures, one method each
internal interface CblasCalls {
    fun dscal(n: Int, alpha: Double, x: DoubleArray, incx: Int)

    fun daxpy(n: Int, alpha: Double, x: DoubleArray, incx: Int, y: DoubleArray, incy: Int)

    fun dgemv(
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
    )

    /** [dgemv] with pointers advanced to logical offsets in the same Kotlin arrays. */
    fun dgemv(
        order: Int,
        trans: Int,
        m: Int,
        n: Int,
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
    ) {
        check(aOffset == 0 && xOffset == 0 && yOffset == 0) { "this CBLAS bridge does not support offsets" }
        dgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }

    fun dger(
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
    )

    fun dsymv(
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
    )

    fun dtrsv(order: Int, uplo: Int, trans: Int, diag: Int, n: Int, a: DoubleArray, lda: Int, x: DoubleArray, incx: Int)

    fun dtrmv(order: Int, uplo: Int, trans: Int, diag: Int, n: Int, a: DoubleArray, lda: Int, x: DoubleArray, incx: Int)

    fun dgemm(
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
    )

    /** [dgemm] with pointers advanced to logical matrix offsets in the same Kotlin arrays. */
    fun dgemm(
        order: Int,
        transA: Int,
        transB: Int,
        m: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        b: DoubleArray,
        bOffset: Int,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
    ) {
        check(aOffset == 0 && bOffset == 0 && cOffset == 0) { "this CBLAS bridge does not support offsets" }
        dgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    fun dsyrk(
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
    )

    fun dsymm(
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
    )

    fun dtrsm(
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
    )

    fun dtrmm(
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
    )
}
