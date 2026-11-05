@file:Suppress("LongParameterList") // BLAS operands, flags and reusable scratch.

package com.eignex.koblas.internal.operations.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.internal.engine.routedWholeCall
import com.eignex.koblas.internal.operations.prepareProduct
import com.eignex.koblas.internal.operations.prepareTriangularMatrix
import com.eignex.koblas.internal.operations.requireGemmOperands
import com.eignex.koblas.internal.operations.requireGemmtOperands
import com.eignex.koblas.internal.operations.requireSymmOperands
import com.eignex.koblas.internal.operations.requireSyr2kOperands
import com.eignex.koblas.internal.operations.requireSyrkOperands
import com.eignex.koblas.internal.operations.triangle
import com.eignex.koblas.internal.scheduling.dense.scheduleGemm
import com.eignex.koblas.internal.scheduling.dense.scheduleGemmt
import com.eignex.koblas.internal.scheduling.dense.scheduleSymm
import com.eignex.koblas.internal.scheduling.dense.scheduleSyr2k
import com.eignex.koblas.internal.scheduling.dense.scheduleSyrk
import com.eignex.koblas.internal.scheduling.dense.scheduleTriangularMatrix
import com.eignex.koblas.internal.storage.sameRegion

internal fun KoblasEngine.executeGemm(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    scratch: Scratch?,
) {
    val depth = if (transposeA) a.rows else a.cols
    val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
    val routed = routedWholeCall(EngineImplementation.Gemm, c.rows, c.cols, depth, borrowed)
    if (routed != null) return routed.gemm(alpha, a, transposeA, b, transposeB, beta, c, scratch)
    requireGemmOperands(a, transposeA, b, transposeB, c)
    prepareProduct(alpha, a, b, beta, c, depth, null, null, null, scratch) { sa, sb ->
        scheduleGemm(alpha, sa, transposeA, sb, transposeB, beta, c, depth, scratch)
    }
}

internal fun KoblasEngine.executeGemmt(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    scratch: Scratch?,
) {
    val depth = if (transposeA) a.rows else a.cols
    val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
    val routed = routedWholeCall(EngineImplementation.Gemmt, c.rows, c.cols, depth, borrowed)
    if (routed != null) return routed.gemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, scratch)
    requireGemmtOperands(a, transposeA, b, transposeB, c)
    prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { sa, sb ->
        scheduleGemmt(alpha, sa, transposeA, sb, transposeB, beta, c, lower, scratch)
    }
}

internal fun KoblasEngine.executeSymm(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    right: Boolean,
    scratch: Scratch?,
) {
    val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
    val routed = routedWholeCall(EngineImplementation.Symm, c.rows, c.cols, a.rows, borrowed)
    if (routed != null) return routed.symm(alpha, a, b, beta, c, lower, right, scratch)
    requireSymmOperands(a, b, c, right)
    // One operand serving both sides needs a full snapshot, since the second reads the triangle the first skips.
    val aTriangle = if (sameRegion(a, b)) null else lower
    prepareProduct(alpha, a, b, beta, c, a.rows, null, aTriangle, this, scratch) { sa, sb ->
        scheduleSymm(alpha, sa, sb, beta, c, lower, right, scratch)
    }
}

internal fun KoblasEngine.executeSyrk(
    alpha: Double,
    a: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    scratch: Scratch?,
) {
    val depth = if (transpose) a.rows else a.cols
    val routed = routedWholeCall(
        EngineImplementation.Syrk,
        c.rows,
        c.cols,
        depth,
        a !is DenseMatrix || c !is DenseMatrix,
    )
    if (routed != null) return routed.syrk(alpha, a, transpose, beta, c, lower, scratch)
    requireSyrkOperands(a, transpose, c)
    prepareProduct(alpha, a, a, beta, c, depth, lower, null, null, scratch) { sa, _ ->
        scheduleSyrk(alpha, sa, transpose, beta, c, lower, scratch)
    }
}

internal fun KoblasEngine.executeSyr2k(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    scratch: Scratch?,
) {
    val depth = if (transpose) a.rows else a.cols
    val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
    val routed = routedWholeCall(EngineImplementation.Syr2k, c.rows, c.cols, depth, borrowed)
    if (routed != null) return routed.syr2k(alpha, a, b, transpose, beta, c, lower, scratch)
    requireSyr2kOperands(a, b, transpose, c)
    prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { sa, sb ->
        scheduleSyr2k(alpha, sa, sb, transpose, beta, c, lower, scratch)
    }
}

internal fun KoblasEngine.executeTrsm(
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    scratch: Scratch?,
) {
    val routed = routedWholeCall(
        EngineImplementation.Trsm,
        b.rows,
        b.cols,
        a.rows,
        a !is DenseMatrix || b !is DenseMatrix,
    )
    if (routed != null) return routed.trsm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
    prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trsm", scratch) { sa ->
        scheduleTriangularMatrix(alpha, sa, triangle(lower, unitDiag), transpose, b, right, true, scratch)
    }
}

internal fun KoblasEngine.executeTrmm(
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    scratch: Scratch?,
) {
    val routed = routedWholeCall(
        EngineImplementation.Trmm,
        b.rows,
        b.cols,
        a.rows,
        a !is DenseMatrix || b !is DenseMatrix,
    )
    if (routed != null) return routed.trmm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
    prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trmm", scratch) { sa ->
        scheduleTriangularMatrix(alpha, sa, triangle(lower, unitDiag), transpose, b, right, false, scratch)
    }
}
