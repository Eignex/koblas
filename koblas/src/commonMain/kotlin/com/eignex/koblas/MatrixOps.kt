@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64MatrixLike
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64VectorLike
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.Uplo
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
public fun F64DenseMatrix.syr(alpha: Double, x: F64VectorLike, uplo: Uplo = Uplo.FULL): Unit = koblas.syr(
    alpha,
    x,
    this,
    uplo,
)

/** Symmetric rank-2 update `A += alpha * (x * yT + y * xT)` (BLAS `dsyr2`) in place. See [F64Blas.syr2]. */
public fun F64DenseMatrix.syr2(alpha: Double, x: F64VectorLike, y: F64VectorLike, uplo: Uplo = Uplo.FULL): Unit =
    koblas.syr2(alpha, x, y, this, uplo)

/**
 * Matrix 1-norm, the maximum absolute column sum (LAPACK `dlange` with norm 1). This is the `anorm`
 * [F64LinearAlgebra.rcond] expects, computed before the matrix is factored.
 */
public fun F64DenseMatrix.norm1(): Double {
    val ad = data
    var m = 0.0
    for (j in 0 until cols) {
        val base = j * rows
        var s = 0.0
        for (i in 0 until rows) s += abs(ad[base + i])
        if (s > m) m = s
    }
    return m
}

/** Matrix infinity-norm, the maximum absolute row sum (LAPACK `dlange` with norm I). */
public fun F64DenseMatrix.normInf(workspace: Workspace? = null): Double {
    if (rows == 0 || cols == 0) return 0.0
    return workspace.scratch(rows) { sums ->
        sums.fill(0.0, 0, rows) // take() promises nothing about the contents
        val ad = data
        for (j in 0 until cols) {
            val base = j * rows
            for (i in 0 until rows) sums[i] += abs(ad[base + i])
        }
        var m = 0.0
        for (i in 0 until rows) if (sums[i] > m) m = sums[i]
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
        if (f != 1.0) koblas.vectorKernels.scale(data, j * rows, f, rows)
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
 * Fresh transposed matrix. For products, prefer the transpose flags on [F64LinearAlgebra.gemv] and
 * [F64LinearAlgebra.gemm], which read the original storage without copying.
 */
public fun F64DenseMatrix.transpose(): F64DenseMatrix {
    val t = F64DenseMatrix(cols, rows)
    val td = t.data
    for (j in 0 until cols) {
        val base = j * rows
        for (i in 0 until rows) td[j + i * cols] = data[base + i]
    }
    return t
}

/**
 * Fresh transposed matrix, still CSC, which makes this the CSC-to-CSR conversion as well. Explicitly
 * stored zeros survive.
 */
public fun F64SparseMatrix.transpose(): F64SparseMatrix {
    val outPtr = IntArray(rows + 1)
    // Counts of row i land in outPtr(i + 1), then a prefix sum turns them into offsets.
    for (k in rowIdx.indices) outPtr[rowIdx[k] + 1]++
    for (i in 0 until rows) outPtr[i + 1] += outPtr[i]
    val outIdx = IntArray(values.size)
    val outVal = DoubleArray(values.size)
    // Walking source columns in order leaves each destination column's row indices ascending.
    val next = outPtr.copyOf()
    for (j in 0 until cols) {
        for (k in colPtr[j] until colPtr[j + 1]) {
            val slot = next[rowIdx[k]]++
            outIdx[slot] = j
            outVal[slot] = values[k]
        }
    }
    return F64SparseMatrix(cols, rows, outPtr, outIdx, outVal)
}

/**
 * Matrix-vector product into a fresh dense result (BLAS `dgemv` with `alpha = 1`, `beta = 0`), for any
 * [F64MatrixLike] against any [F64VectorLike]. No transpose flag; [com.eignex.koblas.sparse.gemv] takes one.
 */
public fun F64MatrixLike.matVec(x: F64VectorLike): F64DenseVector {
    val a = this
    requireShape(a.cols == x.size) { "matVec shape mismatch: A is ${a.rows}x${a.cols}, x size ${x.size}" }
    if (a is F64DenseMatrix && x is F64DenseVector) return F64DenseVector.wrap(koblas.gemv(a, x.data))
    val out = F64DenseVector(a.rows)
    val od = out.data
    if (a is F64DenseMatrix) {
        // One axpy per stored entry of x, down a contiguous column.
        val ad = a.data
        val rows = a.rows
        x.forEachStored { j, v ->
            if (v != 0.0) koblas.vectorKernels.axpy(od, 0, v, ad, j * rows, rows)
        }
    } else if (a is F64SparseMatrix) {
        // Accumulating column j of A scaled by x(j) reads only the stored entries of both operands.
        x.forEachStored { j, v ->
            if (v != 0.0) a.forEachInColumn(j) { i, aij -> od[i] += aij * v }
        }
    } else {
        // A foreign F64MatrixLike is read entry by entry.
        for (i in 0 until a.rows) {
            var s = 0.0
            x.forEachStored { j, v -> s += a[i, j] * v }
            od[i] = s
        }
    }
    return out
}
