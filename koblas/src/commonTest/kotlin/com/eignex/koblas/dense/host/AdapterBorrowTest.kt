package com.eignex.koblas.dense.host

import com.eignex.koblas.Workspace
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.host.CblasCalls
import com.eignex.koblas.dense.host.F64BlasAdapter
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

/**
 * The host adapters borrow scratch around calls into a library that can raise, and a buffer never handed back
 * is one its pool can neither lend again nor reclaim.
 */
class AdapterBorrowTest {

    /**
     * Fails the one entry point the FULL syrk contract needs; nothing else here is reached. A host binding
     * would raise `UnsatisfiedLinkError`, which is not a type common code can name, and what the borrow needs
     * to survive is any throw.
     */
    private class FailingSyrk : CblasCalls {
        override fun dscal(n: Int, alpha: Double, x: DoubleArray, incx: Int) = error("unused")

        override fun daxpy(n: Int, alpha: Double, x: DoubleArray, incx: Int, y: DoubleArray, incy: Int) =
            error("unused")

        @Suppress("LongParameterList")
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
        ) = error("unused")

        @Suppress("LongParameterList")
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
        ) = error("unused")

        @Suppress("LongParameterList")
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
        ) = error("unused")

        @Suppress("LongParameterList")
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
        ) = error("unused")

        @Suppress("LongParameterList")
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
        ) = error("unused")

        @Suppress("LongParameterList")
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
        ) = error("unused")

        @Suppress("LongParameterList")
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
        ): Unit = error("libopenblas.so.0: cannot open shared object file")

        @Suppress("LongParameterList")
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
        ) = error("unused")

        @Suppress("LongParameterList")
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
        ) = error("unused")

        @Suppress("LongParameterList")
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
        ) = error("unused")
    }

    private class FailingAdapter : F64BlasAdapter(FailingSyrk()) {
        override val name: String get() = "failing-syrk"
    }

    @Test
    fun `syrk hands its borrow back when the host library raises`() {
        val n = 64
        val ws = Workspace()
        val parked = ws.take(n * n)
        ws.release(parked)
        assertFailsWith<IllegalStateException> {
            FailingAdapter().syrk(
                1.0,
                F64DenseMatrix(n, n),
                transpose = false,
                beta = 1.0,
                c = F64DenseMatrix(n, n),
                uplo = Uplo.FULL,
                workspace = ws,
            )
        }
        assertSame(parked, ws.take(n * n), "syrk kept its workspace buffer after the host call threw")
    }
}
