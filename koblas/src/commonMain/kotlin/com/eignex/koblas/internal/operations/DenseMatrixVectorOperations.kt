@file:Suppress("LongParameterList") // BLAS operands, flags and optional scratch

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.internal.engine.hostForCall
import com.eignex.koblas.internal.kernels.kernels
import com.eignex.koblas.internal.scheduling.dense.*
import com.eignex.koblas.internal.storage.dense.isWholeBuffer

/** Shared validation and alias preparation supply stable windows to owned and host arithmetic. */
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
    val backend = kernels
    val host = backend.hostForCall(operation, a.rows, a.cols, 1, !a.isWholeBuffer)
    prepareMatrixVector(this, alpha, a, transpose, x, beta, y, null, scratch) { av, ao, ald, values, offset, stride ->
        if (host != null) {
            host.cblasGemv(
                alpha, av, ao, a.rows, a.cols, ald, transpose,
                values, offset, x.size, stride, beta, y.values, y.offset, y.size, y.stride,
            )
        } else {
            backend.scheduleGemv(
                alpha, av, ao, ald, a.rows, a.cols,
                values, offset, stride, beta, y.values, y.offset, y.stride, transpose,
            )
        }
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
    val backend = kernels
    val host = backend.hostForCall(Symv, a.rows, a.cols, 1, !a.isWholeBuffer)
    prepareMatrixVector(this, alpha, a, false, x, beta, y, lower, scratch) { av, ao, ald, values, offset, stride ->
        if (host != null) {
            host.cblasSymv(
                alpha, av, ao, a.rows, a.cols, ald, symmetricStructure(lower),
                values, offset, x.size, stride, beta, y.values, y.offset, y.size, y.stride,
            )
        } else {
            backend.scheduleSymv(
                alpha, av, ao, ald, a.rows, values, offset, stride,
                beta, y.values, y.offset, y.stride, lower,
            )
        }
    }
}

internal fun KoblasEngine.updateDenseRank(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrix,
    scratch: Scratch?,
) {
    val backend = kernels
    val host = backend.hostForCall(Ger, a.rows, a.cols, 1, !a.isWholeBuffer)
    prepareRankUpdate(alpha, x, y, a, RankUpdate.General, scratch) { xv, xo, xs, yv, yo, ys ->
        if (host != null) {
            host.cblasGer(
                alpha, xv, xo, x.size, xs, yv, yo, y.size, ys,
                a.values, a.offset, a.rows, a.cols, a.leadingDimension,
            )
        } else {
            backend.scheduleGer(
                alpha, xv, xo, xs, yv, yo, ys,
                a.values, a.offset, a.leadingDimension, a.rows, a.cols,
            )
        }
    }
}

internal fun KoblasEngine.updateSymmetricRank(
    alpha: Double,
    x: DenseVector,
    a: DenseMatrix,
    lower: Boolean,
    scratch: Scratch?,
) {
    val backend = kernels
    val host = backend.hostForCall(Syr, a.rows, a.rows, 1, !a.isWholeBuffer)
    prepareRankUpdate(alpha, x, x, a, RankUpdate.Symmetric, scratch) { values, offset, stride, _, _, _ ->
        if (host != null) {
            host.cblasSyr(
                alpha, values, offset, x.size, stride,
                a.values, a.offset, a.rows, a.cols, a.leadingDimension, symmetricStructure(lower),
            )
        } else {
            backend.scheduleSyr(alpha, values, offset, stride, a.values, a.offset, a.leadingDimension, a.rows, lower)
        }
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
    val backend = kernels
    val host = backend.hostForCall(Syr2, a.rows, a.rows, 1, !a.isWholeBuffer)
    prepareRankUpdate(alpha, x, y, a, RankUpdate.SymmetricTwo, scratch) { xv, xo, xs, yv, yo, ys ->
        if (host != null) {
            host.cblasSyr2(
                alpha, xv, xo, x.size, xs, yv, yo, y.size, ys,
                a.values, a.offset, a.rows, a.cols, a.leadingDimension, symmetricStructure(lower),
            )
        } else {
            backend.scheduleSyr2(alpha, xv, xo, xs, yv, yo, ys, a.values, a.offset, a.leadingDimension, a.rows, lower)
        }
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
    val backend = kernels
    val host = backend.hostForCall(operation, a.rows, a.rows, 1, !a.isWholeBuffer)
    prepareTriangularVector(a, x, lower, unitDiag, if (solve) "trsv" else "trmv", scratch) { values, offset, lda ->
        if (host != null) {
            host.cblasTriangularVector(
                if (solve) Trsv else Trmv, values, offset, a.rows, a.cols, lda,
                triangle(lower, unitDiag), transpose, x.values, x.offset, x.size, x.stride,
            )
        } else if (solve) {
            backend.scheduleTrsv(values, offset, lda, a.rows, x.values, x.offset, x.stride, lower, transpose, unitDiag)
        } else {
            backend.scheduleTrmv(values, offset, lda, a.rows, x.values, x.offset, x.stride, lower, transpose, unitDiag)
        }
    }
}
