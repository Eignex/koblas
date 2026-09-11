@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("MatrixOpsKt")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

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
public fun MatrixStorage.norm1(): Double {
    var maximum = 0.0
    when (this) {
        is DenseMatrix -> for (j in 0 until cols) {
            maximum = carryingMax(maximum, absoluteSum(data, j * rows, rows))
        }

        is SparseMatrix -> for (j in 0 until cols) {
            val sum = absoluteSum(values, colPtr[j], colPtr[j + 1] - colPtr[j])
            maximum = carryingMax(maximum, sum)
        }
    }
    return maximum
}

/** Matrix infinity-norm, the maximum absolute row sum (LAPACK `dlange` with norm I). A NaN carries through
 *  as it does in [norm1]. */
public fun MatrixStorage.normInf(workspace: Workspace? = null): Double {
    if (rows == 0 || cols == 0) return 0.0
    return workspace.borrow(rows) { sums ->
        sums.fill(0.0, 0, rows) // take() promises nothing about the contents
        when (this) {
            is DenseMatrix -> for (j in 0 until cols) {
                val base = j * rows
                for (i in 0 until rows) sums[i] += abs(data[base + i])
            }

            is SparseMatrix -> for (k in values.indices) {
                sums[rowIdx[k]] += abs(values[k])
            }
        }
        var maximum = 0.0
        for (i in 0 until rows) maximum = carryingMax(maximum, sums[i])
        maximum
    }
}

/** Frobenius norm (LAPACK `dlange` with norm F). Rescales like [norm2] against overflow and underflow. */
public fun MatrixStorage.normFro(): Double {
    val data = when (this) {
        is DenseMatrix -> data
        is SparseMatrix -> values
    }
    return euclideanNorm(data, 0, data.size)
}
