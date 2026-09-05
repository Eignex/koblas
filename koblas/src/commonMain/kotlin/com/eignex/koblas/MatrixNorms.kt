@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("MatrixOpsKt")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

// Part of the MatrixOpsKt facade. Splitting the file would otherwise rename the class JVM callers
// compiled against, so the four parts are joined back into one rather than becoming four.

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.math.abs

/** The larger of [current] and [candidate], except a NaN [candidate] always wins, so it carries through. */
private fun carryingMax(current: Double, candidate: Double): Double =
    if (candidate > current || candidate.isNaN()) candidate else current

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
        m = carryingMax(m, absoluteSum(ad, j * rows, rows))
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
