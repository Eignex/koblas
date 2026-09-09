@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.bench.openblas.*
import kotlinx.cinterop.*

internal actual fun openBlasComparator(): DenseComparator? =
    if (koblas_openblas_available() == 1) NativeOpenBlas else null
internal actual fun oneMklDenseComparator(): DenseComparator? = null
internal actual fun oneMklSparseComparator(): SparseComparator? = null

/** Linux x86-64 OpenBLAS cinterop owned by koblas-bench, including the restored level-1 calls. */
private object NativeOpenBlas : DenseComparator {
    override val identity = "openblas/cblas-native"
    override val threading = "1 thread"

    init { openblas_set_num_threads(1) }

    override fun dot(x: DoubleArray, y: DoubleArray): Double =
        if (x.isEmpty()) 0.0 else x.withPtr { xp -> y.withPtr { yp -> cblas_ddot(x.size, xp, 1, yp, 1) } }

    override fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray) =
        both(x, y) { xp, yp -> cblas_daxpy(x.size, alpha, xp, 1, yp, 1) }
    override fun scale(alpha: Double, x: DoubleArray) = x.withPtr { cblas_dscal(x.size, alpha, it, 1) }
    override fun nrm2(x: DoubleArray): Double = if (x.isEmpty()) 0.0 else x.withPtr { cblas_dnrm2(x.size, it, 1) }
    override fun asum(x: DoubleArray): Double = if (x.isEmpty()) 0.0 else x.withPtr { cblas_dasum(x.size, it, 1) }
    override fun swap(x: DoubleArray, y: DoubleArray) = both(x, y) { xp, yp -> cblas_dswap(x.size, xp, 1, yp, 1) }

    override fun rotm(x: DoubleArray, y: DoubleArray, transformation: ModifiedGivens) {
        if (x.isEmpty() || transformation.flag == -2.0) return
        val p = when (transformation.flag) {
            -1.0 -> doubleArrayOf(-1.0, transformation.h11, transformation.h21, transformation.h12, transformation.h22)
            0.0 -> doubleArrayOf(0.0, 0.0, transformation.h21, transformation.h12, 0.0)
            else -> doubleArrayOf(1.0, transformation.h11, 0.0, 0.0, transformation.h22)
        }
        x.usePinned { xp -> y.usePinned { yp -> p.usePinned { pp -> cblas_drotm(x.size, xp.addressOf(0), 1, yp.addressOf(0), 1, pp.addressOf(0)) } } }
    }

    override fun rot(x: DoubleArray, y: DoubleArray, c: Double, s: Double) =
        both(x, y) { xp, yp -> cblas_drot(x.size, xp, 1, yp, 1, c, s) }

    override fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        three(a.data, x, y) { ap, xp, yp -> cblas_dgemv(COL, trans(transpose), a.rows, a.cols, alpha, ap, a.rows, xp, 1, beta, yp, 1) }
    }
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        three(a.data, x, y) { ap, xp, yp -> cblas_dsymv(COL, uplo(lower), a.rows, alpha, ap, a.rows, xp, 1, beta, yp, 1) }
    }
    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        three(x, y, a.data) { xp, yp, ap -> cblas_dger(COL, a.rows, a.cols, alpha, xp, 1, yp, 1, ap, a.rows) }
    }
    override fun syr(alpha: Double, x: DoubleArray, a: DenseMatrix, lower: Boolean) {
        both(x, a.data) { xp, ap -> cblas_dsyr(COL, uplo(lower), a.rows, alpha, xp, 1, ap, a.rows) }
    }
    override fun syr2(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix, lower: Boolean) {
        three(x, y, a.data) { xp, yp, ap -> cblas_dsyr2(COL, uplo(lower), a.rows, alpha, xp, 1, yp, 1, ap, a.rows) }
    }
    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        both(a.data, x) { ap, xp -> cblas_dtrsv(COL, uplo(lower), trans(transpose), diag(unitDiag), a.rows, ap, a.rows, xp, 1) }
    }
    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        both(a.data, x) { ap, xp -> cblas_dtrmv(COL, uplo(lower), trans(transpose), diag(unitDiag), a.rows, ap, a.rows, xp, 1) }
    }
    override fun gemm(alpha: Double, a: DenseMatrix, transposeA: Boolean, b: DenseMatrix, transposeB: Boolean, beta: Double, c: DenseMatrix) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val n = if (transposeB) b.rows else b.cols
        three(a.data, b.data, c.data) { ap, bp, cp -> cblas_dgemm(COL, trans(transposeA), trans(transposeB), m, n, k, alpha, ap, a.rows, bp, b.rows, beta, cp, c.rows) }
    }
    override fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
    ) {
        three(a.data, b.data, c.data) { ap, bp, cp ->
            cblas_dgemmt(
                COL, uplo(lower), trans(transposeA), trans(transposeB), c.rows,
                if (transposeA) a.rows else a.cols, alpha, ap, a.rows, bp, b.rows, beta, cp, c.rows,
            )
        }
    }
    override fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean) {
        both(a.data, c.data) { ap, cp -> cblas_dsyrk(COL, uplo(lower), trans(transpose), c.rows, if (transpose) a.rows else a.cols, alpha, ap, a.rows, beta, cp, c.rows) }
    }
    override fun syr2k(alpha: Double, a: DenseMatrix, b: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean) {
        three(a.data, b.data, c.data) { ap, bp, cp -> cblas_dsyr2k(COL, uplo(lower), trans(transpose), c.rows, if (transpose) a.rows else a.cols, alpha, ap, a.rows, bp, b.rows, beta, cp, c.rows) }
    }
    override fun symm(alpha: Double, a: DenseMatrix, b: DenseMatrix, beta: Double, c: DenseMatrix, lower: Boolean, right: Boolean) {
        three(a.data, b.data, c.data) { ap, bp, cp -> cblas_dsymm(COL, side(right), uplo(lower), c.rows, c.cols, alpha, ap, a.rows, bp, b.rows, beta, cp, c.rows) }
    }
    override fun trsm(a: DenseMatrix, b: DenseMatrix, lower: Boolean, transpose: Boolean, unitDiag: Boolean, right: Boolean, alpha: Double) {
        both(a.data, b.data) { ap, bp -> cblas_dtrsm(COL, side(right), uplo(lower), trans(transpose), diag(unitDiag), b.rows, b.cols, alpha, ap, a.rows, bp, b.rows) }
    }
    override fun trmm(a: DenseMatrix, b: DenseMatrix, lower: Boolean, transpose: Boolean, unitDiag: Boolean, right: Boolean, alpha: Double) {
        both(a.data, b.data) { ap, bp -> cblas_dtrmm(COL, side(right), uplo(lower), trans(transpose), diag(unitDiag), b.rows, b.cols, alpha, ap, a.rows, bp, b.rows) }
    }
}

private inline fun <R> DoubleArray.withPtr(block: (CPointer<DoubleVar>) -> R): R {
    require(isNotEmpty()) { "empty array has no C address" }
    return usePinned { block(it.addressOf(0)) }
}
private inline fun both(a: DoubleArray, b: DoubleArray, block: (CPointer<DoubleVar>, CPointer<DoubleVar>) -> Unit) {
    if (a.isEmpty() || b.isEmpty()) return
    a.withPtr { ap -> b.withPtr { bp -> block(ap, bp) } }
}
private inline fun three(a: DoubleArray, b: DoubleArray, c: DoubleArray, block: (CPointer<DoubleVar>, CPointer<DoubleVar>, CPointer<DoubleVar>) -> Unit) {
    if (a.isEmpty() || b.isEmpty() || c.isEmpty()) return
    a.withPtr { ap -> b.withPtr { bp -> c.withPtr { cp -> block(ap, bp, cp) } } }
}
private const val COL = 102
private fun trans(v: Boolean) = if (v) 112 else 111
private fun uplo(v: Boolean) = if (v) 122 else 121
private fun diag(v: Boolean) = if (v) 132 else 131
private fun side(v: Boolean) = if (v) 142 else 141
