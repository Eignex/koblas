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

    /**
     * Solve `op(T) · x = b` in place, where `T` is the [lower] or upper triangle of the square [a] and `op`
     * transposes when [transpose] — the sparse `dtrsv`. [x] holds the right-hand side on entry and the
     * solution on return.
     *
     * The solve existed inside [SparseLu] and nowhere else, so a caller holding a triangular `SparseMatrix`
     * of their own had no way to solve against it. This is that routine, on the seam a host sparse BLAS
     * would replace (MKL has `mkl_sparse_d_trsv`; it is one of the routines an inspector-executor interface
     * covers).
     *
     * Every direction walks columns, which is what CSC stores, and only the selected triangle is read — so
     * the other one may hold anything. The two non-transposed forms *push* a finished unknown down or up its
     * column; the transposed forms *pull*, dotting the column against the already-solved entries. That is
     * the same four-way structure the dense cores have, for the same reason: a column of `T` is a row of
     * `Tᵀ`.
     *
     * Unlike the dense cores, this validates the diagonal. `Triangular.kt` leaves a singular dense triangle
     * to produce infinities on the grounds that the caller knows what it passed; here a *structurally*
     * missing diagonal entry is detectable for free while walking the column, so reporting it costs nothing
     * and a silent `NaN` would be a worse trade.
     *
     * @throws IllegalArgumentException if a diagonal entry is missing or zero, naming its position.
     */
    fun trsv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false) {
        require(a.rows == a.cols) { "trsv requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        require(x.size == n) { "trsv: x length ${x.size} != $n" }
        // Column order: forward when a finished unknown feeds later ones, backward when it feeds earlier.
        val order = if (lower != transpose) 0 until n else n - 1 downTo 0
        for (j in order) {
            if (!transpose) {
                // Push: x[j] is final once the earlier columns have subtracted their share.
                val xj = x[j] / diagonalOf(a, j)
                x[j] = xj
                if (xj != 0.0) {
                    a.forEachInColumn(j) { i, v ->
                        // Only the far side of the diagonal is still unknown; the near side is already final.
                        if (if (lower) i > j else i < j) x[i] -= v * xj
                    }
                }
            } else {
                // Pull: column j of T is row j of Tᵀ, so dot it against the entries already solved.
                var s = x[j]
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) s -= v * x[i]
                }
                x[j] = s / diagonalOf(a, j)
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

/**
 * The diagonal entry of column [j], which a triangular solve cannot do without.
 *
 * A structurally absent diagonal is the sparse-specific failure the dense cores have no equivalent of: a
 * dense triangle always *has* the entry, even if it is zero. Both cases are rejected here, and the message
 * distinguishes them because they mean different things to a caller — a missing entry is usually a
 * construction mistake, an explicit zero is usually a singular matrix.
 */
private fun diagonalOf(a: SparseMatrix, j: Int): Double {
    val d = a[j, j]
    require(d != 0.0) {
        val stored = (a.colPtr[j] until a.colPtr[j + 1]).any { a.rowIdx[it] == j }
        if (stored) {
            "trsv: triangle is singular, diagonal entry $j is an explicit zero"
        } else {
            "trsv: triangle has no diagonal entry at $j, so it is singular"
        }
    }
    return d
}
