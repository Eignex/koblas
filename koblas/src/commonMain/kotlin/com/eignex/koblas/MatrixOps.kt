@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.math.abs

/** The larger of [current] and [candidate], except a NaN [candidate] always wins, so it carries through. */
private fun carryingMax(current: Double, candidate: Double): Double =
    if (candidate > current || candidate.isNaN()) candidate else current

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
        m = carryingMax(m, s)
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
        for (i in 0 until rows) m = carryingMax(m, sums[i])
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

/**
 * Scales row `i` by d(i) in place, the product `D * A` for the diagonal D with entries d(i).
 *
 * Runs in `O(nnz)` time, allocates nothing, and keeps the CSC pattern, including explicitly stored zeros,
 * unchanged. The matrix remains mutable through [F64SparseMatrix.values].
 */
public fun F64SparseMatrix.scaleRows(d: DoubleArray) {
    requireShape(d.size == rows) { "scaleRows: d length ${d.size} != $rows rows" }
    for (k in values.indices) values[k] *= d[rowIdx[k]]
}

/**
 * Matrix 1-norm, the maximum absolute column sum (LAPACK `dlange` with norm 1).
 *
 * Runs in `O(nnz + cols)` time and allocates nothing. Explicitly stored zeros contribute zero, while a stored
 * NaN carries through to the result as it does in [F64DenseMatrix.norm1].
 */
public fun F64SparseMatrix.norm1(): Double {
    var maximum = 0.0
    for (j in 0 until cols) {
        val sum = absoluteSum(values, colPtr[j], colPtr[j + 1] - colPtr[j])
        maximum = carryingMax(maximum, sum)
    }
    return maximum
}

/**
 * Matrix infinity-norm, the maximum absolute row sum (LAPACK `dlange` with norm I).
 *
 * Runs in `O(nnz + rows)` time. Without a [Workspace] it allocates a temporary `rows`-element array; a supplied
 * workspace reuses its storage. Explicitly stored zeros contribute zero, and a stored NaN carries through.
 */
public fun F64SparseMatrix.normInf(workspace: Workspace? = null): Double {
    if (rows == 0 || cols == 0) return 0.0
    return workspace.borrow(rows) { sums ->
        sums.fill(0.0, 0, rows)
        for (k in values.indices) sums[rowIdx[k]] += abs(values[k])
        var maximum = 0.0
        for (i in 0 until rows) maximum = carryingMax(maximum, sums[i])
        maximum
    }
}

/**
 * Frobenius norm (LAPACK `dlange` with norm F), rescaled against overflow and underflow like [norm2].
 *
 * Runs in `O(nnz)` time and allocates nothing. Explicitly stored zeros contribute zero; a stored NaN carries
 * through to the result.
 */
public fun F64SparseMatrix.normFro(): Double = euclideanNorm(values, 0, values.size)

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
 * Column [j] as a fresh sparse vector. Runs in `O(nnzⱼ)` time and allocates copies of its stored indices and
 * values, so later mutations to the returned vector's indices or values cannot affect this matrix. Explicitly
 * stored zeros are preserved.
 */
public fun F64SparseMatrix.column(j: Int): F64SparseVector {
    requireIndex(j in 0 until cols) { "column $j outside [0,$cols)" }
    val start = colPtr[j]
    val end = colPtr[j + 1]
    return F64SparseVector.wrap(rows, rowIdx.copyOfRange(start, end), values.copyOfRange(start, end))
}

/**
 * Row [i] as a fresh sparse vector whose stored positions are the source columns. Runs in `O(nnz)` time, scanning
 * every stored entry in the matrix to gather the ones in this row, and allocates arrays sized to its stored
 * entries. The returned vector is independent of this matrix, and explicitly stored zeros are preserved.
 *
 * Extracting every row this way costs `O(nnz * rows)`; use [transpose] once and read its columns instead when
 * the algorithm needs many rows.
 */
public fun F64SparseMatrix.row(i: Int): F64SparseVector {
    requireIndex(i in 0 until rows) { "row $i outside [0,$rows)" }
    var count = 0
    for (j in 0 until cols) {
        for (k in colPtr[j] until colPtr[j + 1]) if (rowIdx[k] == i) count++
    }
    val indices = IntArray(count)
    val out = DoubleArray(count)
    var n = 0
    for (j in 0 until cols) {
        for (k in colPtr[j] until colPtr[j + 1]) {
            if (rowIdx[k] == i) {
                indices[n] = j
                out[n] = values[k]
                n++
            }
        }
    }
    return F64SparseVector.wrap(cols, indices, out)
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
