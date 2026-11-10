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
    withContiguousVector(x, scratch, writeBack = true) { values ->
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
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
        applyBeta(this, y.values, y.offset, y.size, beta, y.stride)
        return
    }
    withContiguousVector(x, scratch) { input ->
        withContiguousVector(y, scratch, read = beta != 0.0, writeBack = true) { output ->
            staged(scratch, input, input === output) { stableX ->
                staged(scratch, a, a.values === output) { stableA -> body(stableA, stableX, output) }
            }
        }
    }
}

/** Packs logical entries only; an overwritten output need not read its previous values. */
private inline fun withContiguousVector(
    vector: DenseVector,
    scratch: Scratch?,
    read: Boolean = true,
    writeBack: Boolean = false,
    body: (DoubleArray) -> Unit,
) {
    if (vector.isWholeBuffer) {
        body(vector.values)
        return
    }
    scratch.borrow(vector.size) { values ->
        if (read) for (i in 0 until vector.size) values[i] = vector[i]
        body(values)
        if (writeBack) for (i in 0 until vector.size) vector[i] = values[i]
    }
}
