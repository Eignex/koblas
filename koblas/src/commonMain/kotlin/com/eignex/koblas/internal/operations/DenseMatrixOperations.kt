@file:Suppress("LongParameterList") // BLAS operands, flags and optional scratch.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.hostForCall
import com.eignex.koblas.internal.kernels.kernels
import com.eignex.koblas.internal.scheduling.dense.*
import com.eignex.koblas.internal.storage.dense.*

/** Shared GEMM validation, alias preparation and arithmetic dispatch. */
internal fun KoblasEngine.multiplyDenseMatrices(
    alpha: Double,
    a: DenseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    scratch: Scratch?,
) {
    val depth = if (transposeA) a.rows else a.cols
    val borrowed = !a.isWholeBuffer || !b.isWholeBuffer || !c.isWholeBuffer
    val backend = kernels
    val routed = backend.hostForCall(EngineImplementation.Gemm, c.rows, c.cols, depth, borrowed)
    requireGemmOperands(a, transposeA, b, transposeB, c)
    prepareProduct(alpha, a, b, beta, c, depth, null, null, null, scratch) { av, ao, ald, bv, bo, bld ->
        if (routed != null) {
            routed.cblasGemm(
                alpha, av, ao, a.rows, a.cols, ald, transposeA,
                bv, bo, b.rows, b.cols, bld, transposeB, beta,
                c.values, c.offset, c.rows, c.cols, c.leadingDimension,
            )
        } else {
            backend.scheduleGemm(
                alpha, av, ao, ald, transposeA, bv, bo, bld, transposeB, depth, beta, c.values, c.offset,
                c.leadingDimension, c.rows, c.cols, scratch,
            )
        }
    }
}

/** Shared selected-triangle product preparation and arithmetic dispatch. */
internal fun KoblasEngine.multiplyDenseMatricesInTriangle(
    alpha: Double,
    a: DenseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    scratch: Scratch?,
) {
    val depth = if (transposeA) a.rows else a.cols
    val borrowed = !a.isWholeBuffer || !b.isWholeBuffer || !c.isWholeBuffer
    val backend = kernels
    val routed = backend.hostForCall(EngineImplementation.Gemmt, c.rows, c.cols, depth, borrowed)
    requireGemmtOperands(a, transposeA, b, transposeB, c)
    prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { av, ao, ald, bv, bo, bld ->
        if (routed != null) {
            routed.cblasGemmt(
                alpha, av, ao, a.rows, a.cols, ald, transposeA,
                bv, bo, b.rows, b.cols, bld, transposeB, beta,
                c.values, c.offset, c.rows, c.cols, c.leadingDimension, symmetricStructure(lower),
            )
        } else {
            backend.scheduleGemmt(
                alpha, av, ao, ald, transposeA, bv, bo, bld, transposeB, depth, beta, c.values, c.offset,
                c.leadingDimension, c.rows, c.cols, lower, scratch,
            )
        }
    }
}

/** Shared symmetric product preparation, including a full snapshot for shared operand mappings. */
internal fun KoblasEngine.multiplySymmetricMatrices(
    alpha: Double,
    a: DenseMatrix,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    right: Boolean,
    scratch: Scratch?,
) {
    val borrowed = !a.isWholeBuffer || !b.isWholeBuffer || !c.isWholeBuffer
    val backend = kernels
    val routed = backend.hostForCall(EngineImplementation.Symm, c.rows, c.cols, a.rows, borrowed)
    requireSymmOperands(a, b, c, right)
    // One operand serving both sides needs a full snapshot, since the second reads the triangle the first skips.
    val aTriangle = if (sameRegion(a, b)) null else lower
    prepareProduct(alpha, a, b, beta, c, a.rows, null, aTriangle, this, scratch) { av, ao, ald, bv, bo, bld ->
        if (routed != null) {
            routed.cblasSymm(
                alpha, av, ao, a.rows, a.cols, ald, symmetricStructure(lower),
                bv, bo, b.rows, b.cols, bld, beta,
                c.values, c.offset, c.rows, c.cols, c.leadingDimension, right,
            )
        } else {
            backend.scheduleSymm(
                alpha, av, ao, ald, a.rows, bv, bo, bld, beta, c.values, c.offset, c.leadingDimension,
                c.rows, c.cols, lower, right, scratch,
            )
        }
    }
}

/** Shared symmetric rank-k preparation and arithmetic dispatch. */
internal fun KoblasEngine.updateSymmetricRankK(
    alpha: Double,
    a: DenseMatrix,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    scratch: Scratch?,
) {
    val depth = if (transpose) a.rows else a.cols
    val backend = kernels
    val routed = backend.hostForCall(
        EngineImplementation.Syrk,
        c.rows,
        c.cols,
        depth,
        !a.isWholeBuffer || !c.isWholeBuffer,
    )
    requireSyrkOperands(a, transpose, c)
    prepareProduct(alpha, a, a, beta, c, depth, lower, null, null, scratch) { values, offset, lda, _, _, _ ->
        if (routed != null) {
            routed.cblasSyrk(
                alpha, values, offset, a.rows, a.cols, lda, transpose, beta,
                c.values, c.offset, c.rows, c.cols, c.leadingDimension, symmetricStructure(lower),
            )
        } else {
            backend.scheduleSyrk(
                alpha, values, offset, lda, transpose, depth, beta, c.values, c.offset, c.leadingDimension,
                c.rows, c.cols, lower, scratch,
            )
        }
    }
}

/** Shared symmetric rank-two-k preparation and arithmetic dispatch. */
internal fun KoblasEngine.updateSymmetricRankTwoK(
    alpha: Double,
    a: DenseMatrix,
    b: DenseMatrix,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    scratch: Scratch?,
) {
    val depth = if (transpose) a.rows else a.cols
    val borrowed = !a.isWholeBuffer || !b.isWholeBuffer || !c.isWholeBuffer
    val backend = kernels
    val routed = backend.hostForCall(EngineImplementation.Syr2k, c.rows, c.cols, depth, borrowed)
    requireSyr2kOperands(a, b, transpose, c)
    prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { av, ao, ald, bv, bo, bld ->
        if (routed != null) {
            routed.cblasSyr2k(
                alpha, av, ao, a.rows, a.cols, ald,
                bv, bo, b.rows, b.cols, bld, transpose, beta,
                c.values, c.offset, c.rows, c.cols, c.leadingDimension, symmetricStructure(lower),
            )
        } else {
            backend.scheduleSyr2k(
                alpha, av, ao, ald, bv, bo, bld, transpose, depth, beta, c.values, c.offset,
                c.leadingDimension, c.rows, c.cols, lower, scratch,
            )
        }
    }
}

/** Shared triangular matrix solve preparation and arithmetic dispatch. */
internal fun KoblasEngine.solveDenseTriangularMatrix(
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    scratch: Scratch?,
) {
    val backend = kernels
    val routed = backend.hostForCall(
        EngineImplementation.Trsm,
        b.rows,
        b.cols,
        a.rows,
        !a.isWholeBuffer || !b.isWholeBuffer,
    )
    prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trsm", scratch) { values, offset, lda ->
        if (routed != null) {
            routed.cblasTriangularMatrix(
                EngineImplementation.Trsm, alpha,
                values, offset, a.rows, a.cols, lda, triangle(lower, unitDiag), transpose,
                b.values, b.offset, b.rows, b.cols, b.leadingDimension, right,
            )
        } else {
            backend.scheduleTriangularMatrix(
                alpha, values, offset, lda, a.rows, triangle(lower, unitDiag), transpose,
                b.values, b.offset, b.leadingDimension, b.rows, b.cols, right, true, scratch,
            )
        }
    }
}

/** Shared triangular matrix multiply preparation and arithmetic dispatch. */
internal fun KoblasEngine.multiplyDenseTriangularMatrix(
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    scratch: Scratch?,
) {
    val backend = kernels
    val routed = backend.hostForCall(
        EngineImplementation.Trmm,
        b.rows,
        b.cols,
        a.rows,
        !a.isWholeBuffer || !b.isWholeBuffer,
    )
    prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trmm", scratch) { values, offset, lda ->
        if (routed != null) {
            routed.cblasTriangularMatrix(
                EngineImplementation.Trmm, alpha,
                values, offset, a.rows, a.cols, lda, triangle(lower, unitDiag), transpose,
                b.values, b.offset, b.rows, b.cols, b.leadingDimension, right,
            )
        } else {
            backend.scheduleTriangularMatrix(
                alpha, values, offset, lda, a.rows, triangle(lower, unitDiag), transpose,
                b.values, b.offset, b.leadingDimension, b.rows, b.cols, right, false, scratch,
            )
        }
    }
}
