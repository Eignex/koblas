@file:Suppress("LongParameterList") // BLAS operands, flags and optional scratch

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.internal.engine.routedWholeCall
import com.eignex.koblas.internal.scheduling.dense.*
import com.eignex.koblas.internal.storage.dense.isWholeBuffer

/** Whole-call dispatch and alias preparation retain views until the numerical schedule begins. */
internal fun KoblasEngine.multiplyMatrixVector(
    alpha: Double,
    a: DenseMatrix,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    transpose: Boolean,
    scratch: Scratch?,
) {
    val operation = if (transpose) GemvTransposed else Gemv
    val host = routedWholeCall(operation, a.rows, a.cols, 1, !a.isWholeBuffer)
    if (host != null) return host.gemv(alpha, a, x, beta, y, transpose, scratch)
    prepareMatrixVector(this, alpha, a, transpose, x, beta, y, null, scratch) { sa, sx ->
        scheduleGemv(alpha, sa, sx.values, sx.offset, sx.stride, beta, y.values, y.offset, y.stride, transpose)
    }
}

internal fun KoblasEngine.multiplySymmetricVector(
    alpha: Double,
    a: DenseMatrix,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    lower: Boolean,
    scratch: Scratch?,
) {
    val host = routedWholeCall(Symv, a.rows, a.cols, 1, !a.isWholeBuffer)
    if (host != null) return host.symv(alpha, a, x, beta, y, lower, scratch)
    prepareMatrixVector(this, alpha, a, false, x, beta, y, lower, scratch) { sa, sx ->
        scheduleSymv(alpha, sa, sx.values, sx.offset, sx.stride, beta, y.values, y.offset, y.stride, lower)
    }
}

internal fun KoblasEngine.updateDenseRank(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrix,
    scratch: Scratch?,
) {
    val host = routedWholeCall(Ger, a.rows, a.cols, 1, !a.isWholeBuffer)
    if (host != null) return host.ger(alpha, x, y, a, scratch)
    prepareRankUpdate(alpha, x, y, a, "ger", scratch) { sx, sy ->
        scheduleGer(
            alpha, sx.values, sx.offset, sx.stride, sy.values, sy.offset, sy.stride,
            a, a.offset, a.rows, a.cols,
        )
    }
}

internal fun KoblasEngine.updateSymmetricRank(
    alpha: Double,
    x: DenseVector,
    a: DenseMatrix,
    lower: Boolean,
    scratch: Scratch?,
) {
    val host = routedWholeCall(Syr, a.rows, a.rows, 1, !a.isWholeBuffer)
    if (host != null) return host.syr(alpha, x, a, lower, scratch)
    prepareRankUpdate(alpha, x, x, a, "syr", scratch) { sx, _ ->
        scheduleSyr(alpha, sx.values, sx.offset, sx.stride, a, lower)
    }
}

internal fun KoblasEngine.updateSymmetricRankTwo(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrix,
    lower: Boolean,
    scratch: Scratch?,
) {
    val host = routedWholeCall(Syr2, a.rows, a.rows, 1, !a.isWholeBuffer)
    if (host != null) return host.syr2(alpha, x, y, a, lower, scratch)
    prepareRankUpdate(alpha, x, y, a, "syr2", scratch) { sx, sy ->
        scheduleSyr2(alpha, sx.values, sx.offset, sx.stride, sy.values, sy.offset, sy.stride, a, lower)
    }
}

internal fun KoblasEngine.applyTriangularVector(
    a: DenseMatrix,
    x: DenseVector,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
    scratch: Scratch?,
) {
    val operation = when {
        solve && transpose -> TrsvTransposed
        solve -> Trsv
        transpose -> TrmvTransposed
        else -> Trmv
    }
    val host = routedWholeCall(operation, a.rows, a.rows, 1, !a.isWholeBuffer)
    if (host != null) {
        if (solve) {
            host.trsv(a, x, lower, transpose, unitDiag, scratch)
        } else {
            host.trmv(a, x, lower, transpose, unitDiag, scratch)
        }
        return
    }
    prepareTriangularVector(a, x, lower, unitDiag, if (solve) "trsv" else "trmv", scratch) { sa ->
        if (solve) {
            scheduleTrsv(sa, x.values, x.offset, x.stride, lower, transpose, unitDiag)
        } else {
            scheduleTrmv(sa, x.values, x.offset, x.stride, lower, transpose, unitDiag)
        }
    }
}
