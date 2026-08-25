@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.dense.host.CblasCalls
import kotlinx.cinterop.*

/**
 * [CblasCalls] over cinterop. Each array is pinned for the duration of its call, so the library reads and
 * writes the Kotlin array in place rather than a copy.
 */
@Suppress("LongParameterList") // the CBLAS signatures
internal class NativeCblasCalls(private val f: CblasFunctions) : CblasCalls {
    override fun dscal(n: Int, alpha: Double, x: DoubleArray, incx: Int) {
        x.usePinned { xp -> f.dscal(n, alpha, xp.addressOf(0), incx) }
    }

    override fun daxpy(n: Int, alpha: Double, x: DoubleArray, incx: Int, y: DoubleArray, incy: Int) {
        x.usePinned { xp -> y.usePinned { yp -> f.daxpy(n, alpha, xp.addressOf(0), incx, yp.addressOf(0), incy) } }
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
        a.usePinned { ap ->
            x.usePinned { xp ->
                y.usePinned { yp ->
                    f.dgemv(
                        order, trans, m, n, alpha,
                        ap.addressOf(
                            0,
                        ),
                        lda, xp.addressOf(0), incx, beta, yp.addressOf(0), incy,
                    )
                }
            }
        }
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
        x.usePinned { xp ->
            y.usePinned { yp ->
                a.usePinned { ap ->
                    f.dger(order, m, n, alpha, xp.addressOf(0), incx, yp.addressOf(0), incy, ap.addressOf(0), lda)
                }
            }
        }
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
        a.usePinned { ap ->
            x.usePinned { xp ->
                y.usePinned { yp ->
                    f.dsymv(
                        order, uplo, n, alpha,
                        ap.addressOf(
                            0,
                        ),
                        lda, xp.addressOf(0), incx, beta, yp.addressOf(0), incy,
                    )
                }
            }
        }
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
        a.usePinned { ap ->
            x.usePinned { xp ->
                f.dtrsv(
                    order, uplo, trans, diag, n,
                    ap.addressOf(
                        0,
                    ),
                    lda, xp.addressOf(0), incx,
                )
            }
        }
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
        a.usePinned { ap ->
            x.usePinned { xp ->
                f.dtrmv(
                    order, uplo, trans, diag, n,
                    ap.addressOf(
                        0,
                    ),
                    lda, xp.addressOf(0), incx,
                )
            }
        }
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
        a.usePinned { ap ->
            b.usePinned { bp ->
                c.usePinned { cp ->
                    f.dgemm(
                        order, transA, transB, m, n, k, alpha,
                        ap.addressOf(0), lda, bp.addressOf(0), ldb, beta, cp.addressOf(0), ldc,
                    )
                }
            }
        }
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
        a.usePinned { ap ->
            c.usePinned { cp ->
                f.dsyrk(order, uplo, trans, n, k, alpha, ap.addressOf(0), lda, beta, cp.addressOf(0), ldc)
            }
        }
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
        a.usePinned { ap ->
            b.usePinned { bp ->
                c.usePinned { cp ->
                    f.dsymm(
                        order, side, uplo, m, n, alpha,
                        ap.addressOf(0), lda, bp.addressOf(0), ldb, beta, cp.addressOf(0), ldc,
                    )
                }
            }
        }
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
        a.usePinned { ap ->
            b.usePinned { bp ->
                f.dtrsm(order, side, uplo, trans, diag, m, n, alpha, ap.addressOf(0), lda, bp.addressOf(0), ldb)
            }
        }
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
        a.usePinned { ap ->
            b.usePinned { bp ->
                f.dtrmm(order, side, uplo, trans, diag, m, n, alpha, ap.addressOf(0), lda, bp.addressOf(0), ldb)
            }
        }
    }
}
