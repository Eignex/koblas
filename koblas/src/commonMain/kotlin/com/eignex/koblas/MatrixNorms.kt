@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.dense.denseMatrixNorm1
import com.eignex.koblas.dense.denseMatrixNormInf
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.sparse.internal.sparseMatrixNorm1
import com.eignex.koblas.sparse.internal.sparseMatrixNormInf

/**
 * Matrix 1-norm, the maximum absolute column sum (LAPACK `dlange` with norm 1). This is the `anorm`
 * rcond expects, computed before the matrix is factored.
 *
 * A NaN entry carries through to the result, as `dlange` carries one through. Comparison alone would drop
 * it, since every comparison against a NaN is false, and a norm that answers a finite number for a matrix
 * it cannot describe would go on to be an `anorm` that hides what is in the matrix.
 */
public fun DenseMatrix.norm1(): Double = denseMatrixNorm1(data, rows, cols)

/** Matrix infinity-norm, the maximum absolute row sum (LAPACK `dlange` with norm I). A NaN carries through
 *  as it does in [norm1]. */
public fun DenseMatrix.normInf(workspace: Workspace? = null): Double {
    if (rows == 0 || cols == 0) return 0.0
    return workspace.borrow(rows) { sums ->
        sums.fill(0.0, 0, rows) // take() promises nothing about the contents
        denseMatrixNormInf(data, rows, cols, sums)
    }
}

/** Frobenius norm (LAPACK `dlange` with norm F). Rescales like [norm2] against overflow and underflow. */
public fun DenseMatrix.normFro(): Double = euclideanNorm(data, 0, data.size)

/**
 * Matrix 1-norm, the maximum absolute column sum (LAPACK `dlange` with norm 1).
 *
 * Runs in `O(nnz + cols)` time and allocates nothing. Explicitly stored zeros contribute zero, while a stored
 * NaN carries through to the result as it does in [DenseMatrix.norm1].
 */
public fun SparseMatrix.norm1(): Double = sparseMatrixNorm1(values, colPtr, cols)

/**
 * Matrix infinity-norm, the maximum absolute row sum (LAPACK `dlange` with norm I).
 *
 * Runs in `O(nnz + rows)` time. Without a [Workspace] it allocates a temporary `rows`-element array; a supplied
 * workspace reuses its storage. Explicitly stored zeros contribute zero, and a stored NaN carries through.
 */
public fun SparseMatrix.normInf(workspace: Workspace? = null): Double {
    if (rows == 0 || cols == 0) return 0.0
    return workspace.borrow(rows) { sums ->
        sums.fill(0.0, 0, rows)
        sparseMatrixNormInf(rowIdx, values, rows, sums)
    }
}

/**
 * Frobenius norm (LAPACK `dlange` with norm F), rescaled against overflow and underflow like [norm2].
 *
 * Runs in `O(nnz)` time and allocates nothing. Explicitly stored zeros contribute zero; a stored NaN carries
 * through to the result.
 */
public fun SparseMatrix.normFro(): Double = euclideanNorm(values, 0, values.size)
