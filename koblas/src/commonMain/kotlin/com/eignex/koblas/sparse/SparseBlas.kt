@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SparseMatrix

/**
 * The sparse matrix routines, the seam a host sparse BLAS plugs into.
 *
 * Thin next to its dense sibling, and honestly so: `gemv` is the one sparse matrix operation koblas has
 * callers for. Sparse `gemm` fills in — the product of two sparse matrices is denser than either, often
 * dramatically — so it is a different algorithm with a different output type, not a flag on this one, and
 * it goes here when something needs it rather than before.
 *
 * The `SparseMatrix` backing crosses to a host library without repacking: its CSC layout is what
 * UMFPACK, KLU and CHOLMOD all consume, and MKL's inspector-executor takes CSC with an index-base flag.
 * See `SparseMatrix` for the verified details and the two remaining mismatches.
 */
interface SparseBlas : Backend {
    /**
     * In-place `y = alpha · op(A) · x + beta · y`, where `op(A)` is `Aᵀ` when [transpose] — the sparse
     * `dgemv`. Per BLAS convention `beta == 0.0` overwrites [y] without reading it, so it may arrive
     * uninitialized, and `alpha == 0.0` reduces to the `beta` scale.
     *
     * Both directions walk the columns, which is what CSC stores. The non-transposed form accumulates
     * `x_j` times column `j` into `y`; the transposed form dots column `j` with `x` to make `y_j`. Neither
     * touches a structural zero.
     */
    @Suppress("LongParameterList") // the BLAS dgemv signature
    fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
    ) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        require(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        require(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        when {
            beta == 0.0 -> y.fill(0.0)
            beta != 1.0 -> for (i in y.indices) y[i] *= beta
        }
        if (alpha == 0.0) return
        if (transpose) {
            for (j in 0 until a.cols) {
                var s = 0.0
                a.forEachInColumn(j) { i, v -> s += v * x[i] }
                y[j] += alpha * s
            }
        } else {
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                if (xj != 0.0) a.forEachInColumn(j) { i, v -> y[i] += v * xj }
            }
        }
    }

    /** `A · x`, or `Aᵀ · x` when [transpose], into a fresh result — the restricted [gemv] with
     *  `alpha = 1, beta = 0`. */
    fun gemv(a: SparseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }
}
