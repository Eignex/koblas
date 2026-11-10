@file:Suppress("LongParameterList") // BLAS operands, flags and scratch

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.scheduling.sparse.*
import com.eignex.koblas.internal.storage.dense.isWholeBuffer

/** Sparse traversal stays portable; the selected engine supplies only numerical leaves. */
internal fun KoblasEngine.multiplySparseMatrixVector(
    alpha: Double,
    a: SparseMatrix,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    transpose: Boolean,
    scratch: Scratch?,
) {
    withStableSparseVectors(alpha, a, x, beta, y, scratch) { sa, sx, sy ->
        scheduleSparseGemv(alpha, sa, sx, beta, sy, transpose)
    }
}

internal fun KoblasEngine.multiplySparseSymmetricVector(
    alpha: Double,
    a: SparseMatrix,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    lower: Boolean,
    scratch: Scratch?,
) {
    withStableSparseVectors(alpha, a, x, beta, y, scratch) { sa, sx, sy ->
        scheduleSparseSymv(alpha, sa, sx, beta, sy, lower)
    }
}

internal fun applySparseTriangularVector(
    a: SparseMatrix,
    x: DenseVector,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
    scratch: Scratch?,
) {
    withContiguousVector(x) { values ->
        staged(scratch, a, a.values === values) { stable ->
            triangularVectorCore(stable, values, lower, transpose, unitDiag, solve)
        }
    }
}

/** Snapshots both inputs before any output scaling, including when all three share one array. */
private inline fun KoblasEngine.withStableSparseVectors(
    alpha: Double,
    a: SparseMatrix,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    scratch: Scratch?,
    body: (SparseMatrix, DoubleArray, DoubleArray) -> Unit,
) {
    withSparseMatrixVectors(alpha, a, x, beta, y) { input, output ->
        staged(scratch, input, input === output) { stableX ->
            staged(scratch, a, a.values === output) { stableA -> body(stableA, stableX, output) }
        }
    }
}

/** Sparse schedules consume contiguous vectors; view packing preserves their no-read and alias rules. */
private inline fun KoblasEngine.withSparseMatrixVectors(
    alpha: Double,
    a: SparseMatrix,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    body: (DoubleArray, DoubleArray) -> Unit,
) {
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
        applyBeta(this, y.values, y.offset, y.size, beta, y.stride)
        return
    }
    val input = if (x.isWholeBuffer) x.values else x.toDoubleArray()
    val output = when {
        y.isWholeBuffer -> y.values
        beta == 0.0 -> DoubleArray(y.size)
        else -> y.toDoubleArray()
    }
    body(input, output)
    if (!y.isWholeBuffer) for (i in 0 until y.size) y[i] = output[i]
}

/** Packs a vector view for an in-place sparse routine and writes its logical entries back. */
private inline fun withContiguousVector(x: DenseVector, body: (DoubleArray) -> Unit) {
    val values = if (x.isWholeBuffer) x.values else x.toDoubleArray()
    body(values)
    if (!x.isWholeBuffer) for (i in 0 until x.size) x[i] = values[i]
}
