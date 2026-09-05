@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.core.F64VectorLike

/*
 * The symmetric rank-one and rank-two updates over a CSC matrix, which build a new pattern rather than
 * writing into an existing one. Their own file beside the other CSC builders rather than inside MatrixOps,
 * where two public extensions sat on top of two hundred lines nothing else could reach.
 */

/** `A + alpha·x·xᵀ` over the [lower] or upper triangle of a CSC [a], as a fresh matrix. */
internal fun sparseSyr(a: F64SparseMatrix, alpha: Double, x: F64VectorLike, lower: Boolean): F64SparseMatrix {
    val xs = x.toDoubleArray()
    if (alpha == 0.0) return a.sparseCopy()
    if (!alpha.isFinite() || xs.any { !it.isFinite() }) return a.syrWithNonFinite(alpha, xs, lower)
    return a.syrFinite(alpha, xs, x.nonzeroSupport(xs), lower)
}

/** `A + alpha·(x·yᵀ + y·xᵀ)` over the [lower] or upper triangle of a CSC [a], as a fresh matrix. */
internal fun sparseSyr2(
    a: F64SparseMatrix,
    alpha: Double,
    x: F64VectorLike,
    y: F64VectorLike,
    lower: Boolean,
): F64SparseMatrix {
    val xs = x.toDoubleArray()
    val ys = y.toDoubleArray()
    if (alpha == 0.0) return a.sparseCopy()
    if (!alpha.isFinite() || xs.any { !it.isFinite() } || ys.any { !it.isFinite() }) {
        return a.syr2WithNonFinite(alpha, xs, ys, lower)
    }
    return a.syr2Finite(alpha, xs, x.nonzeroSupport(xs), ys, y.nonzeroSupport(ys), lower)
}

private fun F64SparseMatrix.sparseCopy(): F64SparseMatrix = F64SparseMatrix.wrap(
    rows,
    cols,
    copyColumnPointers(),
    copyRowIndices(),
    values.copyOf(),
)

private fun F64SparseMatrix.syrFinite(
    alpha: Double,
    x: DoubleArray,
    support: IntArray,
    lower: Boolean,
): F64SparseMatrix {
    val out = SparseRankMatrixBuilder(rows, cols, nnz)
    for (j in 0 until cols) {
        out.beginColumn(j)
        val start = triangleStart(support, j, lower)
        val end = triangleEnd(support, j, lower)
        var source = colPtr[j]
        var update = if (x[j] == 0.0) end else start
        while (source < colPtr[j + 1] || update < end) {
            val sourceRow = if (source < colPtr[j + 1]) rowIdx[source] else Int.MAX_VALUE
            val updateRow = if (update < end) support[update] else Int.MAX_VALUE
            when {
                sourceRow < updateRow -> {
                    out.add(sourceRow, values[source])
                    source++
                }

                updateRow < sourceRow -> {
                    // Adding onto 0.0 rather than storing the bare term keeps an underflowing product at
                    // +0.0, the sign IEEE addition to a zero-initialized entry would produce.
                    out.add(updateRow, 0.0 + (alpha * x[j]) * x[updateRow])
                    update++
                }

                else -> {
                    out.add(sourceRow, values[source] + (alpha * x[j]) * x[sourceRow])
                    source++
                    update++
                }
            }
        }
    }
    return out.build()
}

private fun F64SparseMatrix.syr2Finite(
    alpha: Double,
    x: DoubleArray,
    xSupport: IntArray,
    y: DoubleArray,
    ySupport: IntArray,
    lower: Boolean,
): F64SparseMatrix {
    val out = SparseRankMatrixBuilder(rows, cols, nnz)
    for (j in 0 until cols) {
        out.beginColumn(j)
        val xStart = triangleStart(xSupport, j, lower)
        val xEnd = triangleEnd(xSupport, j, lower)
        val yStart = triangleStart(ySupport, j, lower)
        val yEnd = triangleEnd(ySupport, j, lower)
        var source = colPtr[j]
        var xUpdate = if (y[j] == 0.0) xEnd else xStart
        var yUpdate = if (x[j] == 0.0) yEnd else yStart
        while (source < colPtr[j + 1] || xUpdate < xEnd || yUpdate < yEnd) {
            val sourceRow = if (source < colPtr[j + 1]) rowIdx[source] else Int.MAX_VALUE
            val xRow = if (xUpdate < xEnd) xSupport[xUpdate] else Int.MAX_VALUE
            val yRow = if (yUpdate < yEnd) ySupport[yUpdate] else Int.MAX_VALUE
            val updateRow = minOf(xRow, yRow)
            when {
                sourceRow < updateRow -> {
                    out.add(sourceRow, values[source])
                    source++
                }

                else -> {
                    var value = if (sourceRow == updateRow) values[source++] else 0.0
                    if (xRow == updateRow) {
                        value += (alpha * y[j]) * x[updateRow]
                        xUpdate++
                    }
                    if (yRow == updateRow) {
                        value += (alpha * x[j]) * y[updateRow]
                        yUpdate++
                    }
                    out.add(updateRow, value)
                }
            }
        }
    }
    return out.build()
}

/* Non-finite operands need the dense BLAS visitation order: zero times infinity can itself introduce NaN fill. */
private inline fun F64SparseMatrix.scanTriangleWithFallback(
    lower: Boolean,
    active: (column: Int) -> Boolean,
    contribution: (row: Int, column: Int, current: Double) -> Double,
): F64SparseMatrix {
    val out = SparseRankMatrixBuilder(rows, cols, nnz)
    for (j in 0 until cols) {
        out.beginColumn(j)
        var source = colPtr[j]
        val columnActive = active(j)
        for (i in 0 until rows) {
            val stored = source < colPtr[j + 1] && rowIdx[source] == i
            val selected = if (lower) i >= j else i <= j
            if (selected && columnActive) {
                out.add(i, contribution(i, j, if (stored) values[source] else 0.0))
                if (stored) source++
            } else if (stored) {
                out.add(i, values[source])
                source++
            }
        }
    }
    return out.build()
}

private fun F64SparseMatrix.syrWithNonFinite(alpha: Double, x: DoubleArray, lower: Boolean): F64SparseMatrix =
    scanTriangleWithFallback(lower, active = { j -> x[j] != 0.0 }) { i, j, current ->
        current + (alpha * x[j]) * x[i]
    }

private fun F64SparseMatrix.syr2WithNonFinite(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    lower: Boolean,
): F64SparseMatrix = scanTriangleWithFallback(lower, active = { j -> x[j] != 0.0 || y[j] != 0.0 }) { i, j, current ->
    var value = current
    value += (alpha * y[j]) * x[i]
    value += (alpha * x[j]) * y[i]
    value
}

/**
 * Ascending indices where this vector is genuinely nonzero, for driving the sparse `syr`/`syr2` merge. A
 * [F64SparseVector] already knows its stored positions, so it filters those instead of paying to rediscover
 * them by rescanning [dense], the already-materialized copy of this vector. [filterNonzero] only reads
 * [F64SparseVector.indices], and hands the same live array back unchanged when nothing needs filtering, so
 * the common case of a sparse vector with no explicit zeros costs no copy at all.
 */
@OptIn(UnsafeKoblasApi::class)
private fun F64VectorLike.nonzeroSupport(dense: DoubleArray): IntArray = when (this) {
    is F64SparseVector -> filterNonzero(indices, values)
    else -> nonzeroIndices(dense)
}

private fun filterNonzero(indices: IntArray, values: DoubleArray): IntArray {
    var count = 0
    for (value in values) if (value != 0.0) count++
    if (count == values.size) return indices
    val out = IntArray(count)
    var at = 0
    for (k in values.indices) if (values[k] != 0.0) out[at++] = indices[k]
    return out
}

private fun nonzeroIndices(values: DoubleArray): IntArray {
    var count = 0
    for (value in values) if (value != 0.0) count++
    val out = IntArray(count)
    var at = 0
    for (i in values.indices) if (values[i] != 0.0) out[at++] = i
    return out
}

private fun triangleStart(indices: IntArray, column: Int, lower: Boolean): Int = if (lower) {
    lowerBound(indices, column)
} else {
    0
}

private fun triangleEnd(indices: IntArray, column: Int, lower: Boolean): Int = if (lower) {
    indices.size
} else {
    lowerBound(indices, column + 1)
}

private fun lowerBound(indices: IntArray, value: Int): Int {
    var low = 0
    var high = indices.size
    while (low < high) {
        val middle = (low + high) ushr 1
        if (indices[middle] < value) low = middle + 1 else high = middle
    }
    return low
}

private class SparseRankMatrixBuilder(private val rows: Int, private val cols: Int, expectedEntries: Int) {
    private val pointers = IntArray(cols + 1)
    private var rowIndices = IntArray(expectedEntries)
    private var coefficients = DoubleArray(expectedEntries)
    private var size = 0

    fun beginColumn(column: Int) {
        pointers[column] = size
    }

    fun add(row: Int, value: Double) {
        if (size == rowIndices.size) grow()
        rowIndices[size] = row
        coefficients[size] = value
        size++
    }

    fun build(): F64SparseMatrix {
        pointers[cols] = size
        return F64SparseMatrix.wrap(rows, cols, pointers, rowIndices.copyOf(size), coefficients.copyOf(size))
    }

    private fun grow() {
        val next = if (rowIndices.isEmpty()) 4 else rowIndices.size * 2
        rowIndices = rowIndices.copyOf(next)
        coefficients = coefficients.copyOf(next)
    }
}
