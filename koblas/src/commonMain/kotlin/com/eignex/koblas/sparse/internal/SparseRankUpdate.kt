@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Vector

/*
 * The symmetric rank-one and rank-two updates over a CSC matrix, which build a new pattern rather than
 * writing into an existing one. Their own file beside the other CSC builders rather than inside MatrixOps,
 * where two public extensions sat on top of two hundred lines nothing else could reach.
 */

/** `A + alpha·x·xᵀ` over the [lower] or upper triangle of a CSC [a], as a fresh matrix. */
internal fun sparseSyr(a: SparseMatrix, alpha: Double, x: Vector, lower: Boolean): SparseMatrix {
    val xs = x.toDoubleArray()
    if (alpha == 0.0) return a.sparseCopy()
    if (!alpha.isFinite() || xs.any { !it.isFinite() }) return a.syrWithNonFinite(alpha, xs, lower)
    return a.syrFinite(alpha, xs, x.nonzeroSupport(xs), lower)
}

/** `A + alpha·(x·yᵀ + y·xᵀ)` over the [lower] or upper triangle of a CSC [a], as a fresh matrix. */
internal fun sparseSyr2(a: SparseMatrix, alpha: Double, x: Vector, y: Vector, lower: Boolean): SparseMatrix {
    val xs = x.toDoubleArray()
    val ys = y.toDoubleArray()
    if (alpha == 0.0) return a.sparseCopy()
    if (!alpha.isFinite() || xs.any { !it.isFinite() } || ys.any { !it.isFinite() }) {
        return a.syr2WithNonFinite(alpha, xs, ys, lower)
    }
    return a.syr2Finite(alpha, xs, x.nonzeroSupport(xs), ys, y.nonzeroSupport(ys), lower)
}

private fun SparseMatrix.sparseCopy(): SparseMatrix = SparseMatrix.wrap(
    rows,
    cols,
    copyColumnPointers(),
    copyRowIndices(),
    values.copyOf(),
)

private fun SparseMatrix.syrFinite(alpha: Double, x: DoubleArray, support: IntArray, lower: Boolean): SparseMatrix {
    val out = SparseRankMatrixBuilder(rows, cols, nnz)
    for (j in 0 until cols) {
        out.beginColumn(j)
        val start = triangleStart(support, j, lower)
        val end = triangleEnd(support, j, lower)
        val updateStart = if (x[j] == 0.0) end else start
        out.appendRankOne(alpha, j, x, support, updateStart, end, this)
    }
    return out.build()
}

private fun SparseMatrix.syr2Finite(
    alpha: Double,
    x: DoubleArray,
    xSupport: IntArray,
    y: DoubleArray,
    ySupport: IntArray,
    lower: Boolean,
): SparseMatrix {
    val out = SparseRankMatrixBuilder(rows, cols, nnz)
    for (j in 0 until cols) {
        out.beginColumn(j)
        val xStart = triangleStart(xSupport, j, lower)
        val xEnd = triangleEnd(xSupport, j, lower)
        val yStart = triangleStart(ySupport, j, lower)
        val yEnd = triangleEnd(ySupport, j, lower)
        out.appendRankTwo(
            alpha, j, x, xSupport, if (y[j] == 0.0) xEnd else xStart, xEnd,
            y, ySupport, if (x[j] == 0.0) yEnd else yStart, yEnd, this,
        )
    }
    return out.build()
}

/* Non-finite operands need the dense BLAS visitation order: zero times infinity can itself introduce NaN fill. */
private fun SparseMatrix.syrWithNonFinite(alpha: Double, x: DoubleArray, lower: Boolean): SparseMatrix {
    val out = SparseRankMatrixBuilder(rows, cols, nnz)
    for (j in 0 until cols) {
        out.beginColumn(j)
        out.appendRankOneDense(alpha, j, x, lower, this)
    }
    return out.build()
}

private fun SparseMatrix.syr2WithNonFinite(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    lower: Boolean,
): SparseMatrix {
    val out = SparseRankMatrixBuilder(rows, cols, nnz)
    for (j in 0 until cols) {
        out.beginColumn(j)
        out.appendRankTwoDense(alpha, j, x, y, lower, this)
    }
    return out.build()
}

/**
 * Ascending indices where this vector is genuinely nonzero, for driving the sparse `syr`/`syr2` merge. A
 * [SparseVector] already knows its stored positions, so it filters those instead of paying to rediscover
 * them by rescanning [dense], the already-materialized copy of this vector. [filterNonzero] only reads
 * [SparseVector.indices], and hands the same live array back unchanged when nothing needs filtering, so
 * the common case of a sparse vector with no explicit zeros costs no copy at all.
 */
@OptIn(UnsafeKoblasApi::class)
private fun Vector.nonzeroSupport(dense: DoubleArray): IntArray = when (this) {
    is SparseVector -> filterNonzero(indices, values)
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

    @Suppress("LongParameterList")
    fun appendRankOne(
        alpha: Double,
        column: Int,
        x: DoubleArray,
        support: IntArray,
        supportStart: Int,
        supportEnd: Int,
        source: SparseMatrix,
    ) {
        ensureCapacity(
            (source.colPtr[column + 1] - source.colPtr[column]).toLong() + supportEnd - supportStart,
        )
        size += SparseAccumulationKernels.mergeRankOneColumn(
            alpha, column, x, support, supportStart, supportEnd,
            source.rowIdx, source.values, source.colPtr[column], source.colPtr[column + 1],
            rowIndices, coefficients, size,
        )
    }

    @Suppress("LongParameterList")
    fun appendRankTwo(
        alpha: Double,
        column: Int,
        x: DoubleArray,
        xSupport: IntArray,
        xStart: Int,
        xEnd: Int,
        y: DoubleArray,
        ySupport: IntArray,
        yStart: Int,
        yEnd: Int,
        source: SparseMatrix,
    ) {
        ensureCapacity(
            (source.colPtr[column + 1] - source.colPtr[column]).toLong() +
                (xEnd - xStart).toLong() + yEnd - yStart,
        )
        size += SparseAccumulationKernels.mergeRankTwoColumn(
            alpha, column, x, xSupport, xStart, xEnd, y, ySupport, yStart, yEnd,
            source.rowIdx, source.values, source.colPtr[column], source.colPtr[column + 1],
            rowIndices, coefficients, size,
        )
    }

    fun appendRankOneDense(alpha: Double, column: Int, x: DoubleArray, lower: Boolean, source: SparseMatrix) {
        ensureCapacity(rows.toLong())
        size += SparseAccumulationKernels.updateRankOneDenseColumn(
            alpha, column, x, lower, rows,
            source.rowIdx, source.values, source.colPtr[column], source.colPtr[column + 1],
            rowIndices, coefficients, size,
        )
    }

    fun appendRankTwoDense(
        alpha: Double,
        column: Int,
        x: DoubleArray,
        y: DoubleArray,
        lower: Boolean,
        source: SparseMatrix,
    ) {
        ensureCapacity(rows.toLong())
        size += SparseAccumulationKernels.updateRankTwoDenseColumn(
            alpha, column, x, y, lower, rows,
            source.rowIdx, source.values, source.colPtr[column], source.colPtr[column + 1],
            rowIndices, coefficients, size,
        )
    }

    fun build(): SparseMatrix {
        pointers[cols] = size
        return SparseMatrix.wrap(rows, cols, pointers, rowIndices.copyOf(size), coefficients.copyOf(size))
    }

    private fun ensureCapacity(additional: Long) {
        val required = size.toLong() + additional
        if (required <= rowIndices.size) return
        require(required <= Int.MAX_VALUE) { "sparse rank update exceeds array capacity" }
        val next = maxOf(if (rowIndices.isEmpty()) 4L else rowIndices.size.toLong() * 2, required)
            .coerceAtMost(Int.MAX_VALUE.toLong())
            .toInt()
        rowIndices = rowIndices.copyOf(next)
        coefficients = coefficients.copyOf(next)
    }
}
