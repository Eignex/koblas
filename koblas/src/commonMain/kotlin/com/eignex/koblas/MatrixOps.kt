@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.math.abs

/**
 * Rank-one update `A = A + alpha * x * yT` (BLAS `dger`) in place. Subtract by passing
 * `alpha = -1.0`.
 */
public fun F64DenseMatrix.ger(alpha: Double, x: F64VectorLike, y: F64VectorLike) {
    requireShape(rows == x.size && cols == y.size) {
        "ger shape mismatch: A is ${rows}x$cols, x ${x.size}, y ${y.size}"
    }
    if (alpha == 0.0) return
    if (x is F64DenseVector && y is F64DenseVector) {
        koblas.ger(alpha, x.data, y.data, this)
        return
    }
    val md = data
    y.forEachStored { j, yj ->
        if (yj != 0.0) {
            val col = j * rows
            val scaled = alpha * yj
            x.forEachStored { i, xi -> md[col + i] += scaled * xi }
        }
    }
}

/** Symmetric rank-1 update `A += alpha * x * xT` (BLAS `dsyr`) in place. See [F64Blas.syr]. */
public fun F64DenseMatrix.syr(alpha: Double, x: F64VectorLike, lower: Boolean = true): Unit = koblas.syr(
    alpha,
    x,
    this,
    lower,
)

/** Symmetric rank-2 update `A += alpha * (x * yT + y * xT)` (BLAS `dsyr2`) in place. See [F64Blas.syr2]. */
public fun F64DenseMatrix.syr2(alpha: Double, x: F64VectorLike, y: F64VectorLike, lower: Boolean = true): Unit =
    koblas.syr2(alpha, x, y, this, lower)

/**
 * Fresh CSC matrix holding `A + alpha * x * xT` in its [lower] or upper triangle. The other triangle is
 * copied unchanged. Unlike dense [F64DenseMatrix.syr], this is not in place: a rank update can introduce
 * entries that the source CSC pattern has no room to store.
 *
 * Existing explicit zeros survive. A coordinate reached by nonzero vector support is stored even when its
 * arithmetic cancels or underflows to zero, so the returned matrix never silently drops discovered fill.
 * The result owns independent structural and value arrays, and its rows ascend within every column.
 */
public fun F64SparseMatrix.syr(alpha: Double, x: F64VectorLike, lower: Boolean = true): F64SparseMatrix {
    requireShape(rows == cols) { "syr: matrix must be square, got ${rows}x$cols" }
    requireShape(x.size == rows) { "syr: x length ${x.size} != $rows" }
    val xs = x.toDoubleArray()
    if (alpha == 0.0) return sparseCopy()
    if (!alpha.isFinite() || xs.any { !it.isFinite() }) return syrWithNonFinite(alpha, xs, lower)
    return syrFinite(alpha, xs, x.nonzeroSupport(xs), lower)
}

/**
 * Fresh CSC matrix holding `A + alpha * (x * yT + y * xT)` in its [lower] or upper triangle. The other
 * triangle is copied unchanged. This structural counterpart of dense [F64DenseMatrix.syr2] never mutates
 * its source, because newly nonzero entries may require CSC fill.
 *
 * Existing explicit zeros survive. A coordinate reached by nonzero vector support is stored even when its
 * two terms cancel or underflow to zero. The result has independent arrays and canonical ascending CSC rows.
 */
public fun F64SparseMatrix.syr2(
    alpha: Double,
    x: F64VectorLike,
    y: F64VectorLike,
    lower: Boolean = true,
): F64SparseMatrix {
    requireShape(rows == cols) { "syr2: matrix must be square, got ${rows}x$cols" }
    requireShape(x.size == rows && y.size == rows) {
        "syr2: operand lengths ${x.size} and ${y.size} must both be $rows"
    }
    val xs = x.toDoubleArray()
    val ys = y.toDoubleArray()
    if (alpha == 0.0) return sparseCopy()
    if (!alpha.isFinite() || xs.any { !it.isFinite() } || ys.any { !it.isFinite() }) {
        return syr2WithNonFinite(alpha, xs, ys, lower)
    }
    return syr2Finite(alpha, xs, x.nonzeroSupport(xs), ys, y.nonzeroSupport(ys), lower)
}

/**
 * Matrix 1-norm, the maximum absolute column sum (LAPACK `dlange` with norm 1). This is the `anorm`
 * rcond expects, computed before the matrix is factored.
 *
 * A NaN entry carries through to the result, as `dlange` carries one through. Comparison alone would drop
 * it, since every comparison against a NaN is false, and a norm that answers a finite number for a matrix
 * it cannot describe would go on to be an `anorm` that hides what is in the matrix.
 */
public fun F64DenseMatrix.norm1(): Double {
    val ad = data
    var m = 0.0
    for (j in 0 until cols) {
        val base = j * rows
        var s = 0.0
        for (i in 0 until rows) s += abs(ad[base + i])
        if (s > m || s.isNaN()) m = s
    }
    return m
}

/** Matrix infinity-norm, the maximum absolute row sum (LAPACK `dlange` with norm I). A NaN carries through
 *  as it does in [norm1]. */
public fun F64DenseMatrix.normInf(workspace: Workspace? = null): Double {
    if (rows == 0 || cols == 0) return 0.0
    return workspace.borrow(rows) { sums ->
        sums.fill(0.0, 0, rows) // take() promises nothing about the contents
        val ad = data
        for (j in 0 until cols) {
            val base = j * rows
            for (i in 0 until rows) sums[i] += abs(ad[base + i])
        }
        var m = 0.0
        for (i in 0 until rows) if (sums[i] > m || sums[i].isNaN()) m = sums[i]
        m
    }
}

/** Frobenius norm (LAPACK `dlange` with norm F). Rescales like [norm2] against overflow and underflow. */
public fun F64DenseMatrix.normFro(): Double = euclideanNorm(data, 0, data.size)

/** Scale row `i` by d(i) in place, the product `D * A` for the diagonal D with entries d(i). */
public fun F64DenseMatrix.scaleRows(d: DoubleArray) {
    requireShape(d.size == rows) { "scaleRows: d length ${d.size} != $rows rows" }
    val ad = data
    for (j in 0 until cols) {
        val base = j * rows
        for (i in 0 until rows) ad[base + i] *= d[i]
    }
}

/** Scale column `j` by d(j) in place, the product `A * D` for the diagonal D with entries d(j). */
public fun F64DenseMatrix.scaleColumns(d: DoubleArray) {
    requireShape(d.size == cols) { "scaleColumns: d length ${d.size} != $cols columns" }
    for (j in 0 until cols) {
        val f = d[j]
        if (f != 1.0) koblas.kernels.scale(data, j * rows, f, rows)
    }
}

/** Scale column `j` by d(j) in place for a CSC matrix. The pattern is untouched. */
public fun F64SparseMatrix.scaleColumns(d: DoubleArray) {
    requireShape(d.size == cols) { "scaleColumns: d length ${d.size} != $cols columns" }
    for (j in 0 until cols) {
        val f = d[j]
        if (f == 1.0) continue
        for (k in colPtr[j] until colPtr[j + 1]) values[k] *= f
    }
}

/** Column `j` as a fresh vector, copied rather than viewed. */
public fun F64DenseMatrix.column(j: Int): F64DenseVector {
    requireIndex(j in 0 until cols) { "column $j outside [0,$cols)" }
    val start = j * rows
    return F64DenseVector.wrap(data.copyOfRange(start, start + rows))
}

/** Row `i` as a fresh vector, gathered across the backing. Prefer [column] where the algorithm allows. */
public fun F64DenseMatrix.row(i: Int): F64DenseVector {
    requireIndex(i in 0 until rows) { "row $i outside [0,$rows)" }
    val out = DoubleArray(cols)
    for (j in 0 until cols) out[j] = data[i + j * rows]
    return F64DenseVector.wrap(out)
}

/**
 * Fresh transposed matrix, with the active backend ([koblas]). For products, prefer the transpose flags on
 * gemv and gemm, which read the original storage without copying. See [F64Blas.transpose].
 */
public fun F64DenseMatrix.transpose(): F64DenseMatrix = koblas.blas.transpose(this)

/**
 * Fresh transposed matrix, still CSC, which makes this the CSC-to-CSR conversion as well, with the active
 * backend ([koblas]). See [com.eignex.koblas.sparse.F64SparseBlas.transpose].
 */
public fun F64SparseMatrix.transpose(): F64SparseMatrix = koblas.sparseBlas.transpose(this)

/**
 * Fresh matrix with column [column] replaced by [entering], still CSC. The replacement is structural, so an
 * explicitly stored zero in [entering] survives as one.
 */
public fun F64SparseMatrix.withColumn(column: Int, entering: F64SparseVector): F64SparseMatrix {
    requireIndex(column in 0 until cols) { "withColumn: column $column outside [0,$cols)" }
    requireShape(entering.size == rows) { "withColumn: entering size ${entering.size}, expected $rows" }
    val start = colPtr[column]
    val end = colPtr[column + 1]
    val delta = entering.indices.size - (end - start)
    val pointers = IntArray(cols + 1) { colPtr[it] + if (it <= column) 0 else delta }
    val outIdx = IntArray(rowIdx.size + delta)
    val outVal = DoubleArray(values.size + delta)
    rowIdx.copyInto(outIdx, endIndex = start)
    values.copyInto(outVal, endIndex = start)
    entering.indices.copyInto(outIdx, start)
    entering.values.copyInto(outVal, start)
    rowIdx.copyInto(outIdx, start + entering.indices.size, end)
    values.copyInto(outVal, start + entering.indices.size, end)
    return F64SparseMatrix(rows, cols, pointers, outIdx, outVal)
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
