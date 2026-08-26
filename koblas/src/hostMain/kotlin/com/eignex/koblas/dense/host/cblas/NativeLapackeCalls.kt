@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.dense.host.LapackeCalls
import kotlinx.cinterop.*

/** [LapackeCalls] over cinterop, alongside [NativeCblasCalls]. */
@Suppress("LongParameterList") // the LAPACKE signatures
internal class NativeLapackeCalls(private val f: LapackeFunctions) : LapackeCalls {
    override fun dgetrf(order: Int, m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        a.usePinned { ap -> ipiv.usePinned { ip -> f.dgetrf(order, m, n, ap.addressOf(0), lda, ip.addressOf(0)) } }

    override fun dgecon(
        order: Int,
        norm: Byte,
        n: Int,
        a: DoubleArray,
        lda: Int,
        anorm: Double,
        rcond: DoubleArray,
    ): Int = a.usePinned { ap ->
        rcond.usePinned { rp -> f.dgecon(order, norm, n, ap.addressOf(0), lda, anorm, rp.addressOf(0)) }
    }

    override fun dgeqrf(order: Int, m: Int, n: Int, a: DoubleArray, lda: Int, tau: DoubleArray): Int =
        a.usePinned { ap -> tau.usePinned { tp -> f.dgeqrf(order, m, n, ap.addressOf(0), lda, tp.addressOf(0)) } }

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
    ): Int = a.usePinned { ap ->
        tau.usePinned { tp ->
            c.usePinned { cp ->
                f.dormqr(order, side, trans, m, n, k, ap.addressOf(0), lda, tp.addressOf(0), cp.addressOf(0), ldc)
            }
        }
    }

    override fun dpotrf(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int): Int =
        a.usePinned { ap -> f.dpotrf(order, uplo, n, ap.addressOf(0), lda) }

    override fun dgetri(order: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        a.usePinned { ap -> ipiv.usePinned { ip -> f.dgetri(order, n, ap.addressOf(0), lda, ip.addressOf(0)) } }

    override fun dtrtri(order: Int, uplo: Byte, diag: Byte, n: Int, a: DoubleArray, lda: Int): Int =
        a.usePinned { ap -> f.dtrtri(order, uplo, diag, n, ap.addressOf(0), lda) }

    override fun dpotri(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int): Int =
        a.usePinned { ap -> f.dpotri(order, uplo, n, ap.addressOf(0), lda) }

    override fun dsytrf(order: Int, uplo: Byte, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        a.usePinned { ap -> ipiv.usePinned { ip -> f.dsytrf(order, uplo, n, ap.addressOf(0), lda, ip.addressOf(0)) } }

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
    ): Int = a.usePinned { ap ->
        ipiv.usePinned { ip ->
            b.usePinned { bp ->
                f.dsytrs(order, uplo, n, nrhs, ap.addressOf(0), lda, ip.addressOf(0), bp.addressOf(0), ldb)
            }
        }
    }
}
