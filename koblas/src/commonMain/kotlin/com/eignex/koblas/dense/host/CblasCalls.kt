package com.eignex.koblas.dense.host

/**
 * The CBLAS entry points koblas binds, in plain arrays and ints. The two host bindings differ only in how
 * they hand an array to the library, so this is the seam between that mechanism and the routines built on
 * it: the JVM implements it over `java.lang.foreign`, Kotlin/Native over `usePinned`.
 *
 * Every array is passed whole and read from its start; leading dimensions come as separate arguments, as in
 * CBLAS itself. An implementation must not copy, since [F64BlasAdapter] relies on the library writing
 * through to the caller's arrays.
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
