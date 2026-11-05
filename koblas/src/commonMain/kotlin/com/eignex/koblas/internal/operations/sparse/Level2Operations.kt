@file:Suppress("LongParameterList") // BLAS operands, flags and reusable scratch.

package com.eignex.koblas.internal.operations.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.operations.applyBeta
import com.eignex.koblas.internal.operations.requireGemvOperands
import com.eignex.koblas.internal.operations.requireSymvOperands
import com.eignex.koblas.internal.operations.requireTriangularVectorOperands
import com.eignex.koblas.internal.operations.staged
import com.eignex.koblas.internal.scheduling.sparse.scheduleSparseGemv
import com.eignex.koblas.internal.scheduling.sparse.scheduleSparseSymv
import com.eignex.koblas.internal.scheduling.sparse.triangularVectorCore

internal fun KoblasEngine.executeGemv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean,
    scratch: Scratch?,
) {
    requireGemvOperands(a, transpose, x.size, y.size)
    // Zero alpha or zero extent reads neither operand; only destination scaling remains.
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return applyBeta(this, y, 0, y.size, beta)
    // Snapshot aliased operands before destination scaling overwrites their coefficients.
    staged(scratch, x, x === y) { stableX ->
        staged(
            scratch,
            a,
            a.values === y,
        ) { stableA -> scheduleSparseGemv(alpha, stableA, stableX, beta, y, transpose) }
    }
}

internal fun KoblasEngine.executeSymv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean,
    scratch: Scratch?,
) {
    requireSymvOperands(a, x.size, y.size)
    if (alpha == 0.0) return applyBeta(this, y, 0, y.size, beta)
    staged(scratch, x, x === y) { stableX ->
        staged(
            scratch,
            a,
            a.values === y,
        ) { stableA -> scheduleSparseSymv(alpha, stableA, stableX, beta, y, lower) }
    }
}

internal fun KoblasEngine.executeTrsv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    scratch: Scratch?,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    // Snapshot an aliased triangle before substitution overwrites coefficients needed by later columns.
    staged(scratch, a, a.values === x) { stable ->
        triangularVectorCore(stable, x, lower, transpose, unitDiag, solve = true)
    }
}

internal fun KoblasEngine.executeTrmv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    scratch: Scratch?,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    staged(scratch, a, a.values === x) { stable ->
        triangularVectorCore(stable, x, lower, transpose, unitDiag, solve = false)
    }
}
