@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, T, X

package com.eignex.koblas

// Triangular solves (BLAS dtrsv / dtrsm) over the row-major flat backing of [DenseMatrix].
//
// Only the triangle selected by `lower` is ever read — the opposite triangle may hold anything, so
// the packed LU buffer (unit-lower L below the diagonal, U on and above) and a Cholesky factor both
// work as inputs directly. Row-oriented substitutions reduce to [denseDot] on contiguous row runs;
// the transposed direction is column-oriented — once an unknown is final, its contribution is
// subtracted from the remaining right-hand side along a contiguous row via [denseAxpy].
//
// Following BLAS, the diagonal is not checked: a zero (or, with `unitDiag`, implicit-one) diagonal
// is the caller's responsibility, and a singular triangle yields infinities/NaNs, not an exception.

/**
 * Solve `op(T) · x = b` in place (BLAS `dtrsv`): [x] holds `b` on entry and the solution on return.
 * `T` is the [lower] or upper triangle of the square [A]; `op` transposes when [transpose]; with
 * [unitDiag] the diagonal is taken as 1 and never read.
 */
fun trsv(A: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) {
    require(A.rows == A.cols) { "trsv requires a square matrix; got ${A.rows}x${A.cols}" }
    require(x.size == A.rows) { "trsv: x length ${x.size} != ${A.rows}" }
    trsvCore(A.data, A.rows, x, lower, transpose, unitDiag)
}

/**
 * Solve `op(T) · X = B` in place for [nrhs][DenseMatrix.cols] right-hand sides (BLAS `dtrsm`,
 * left side only): [B] holds the right-hand sides column-wise on entry and the solutions on return.
 * `T` is the [lower] or upper triangle of the square [A]; `op` transposes when [transpose]; with
 * [unitDiag] the diagonal is taken as 1 and never read. BLAS `side = R` (`X · op(T) = B`) is out of
 * the supported subset.
 */
fun trsm(A: DenseMatrix, B: DenseMatrix, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) {
    require(A.rows == A.cols) { "trsm requires a square matrix; got ${A.rows}x${A.cols}" }
    require(B.rows == A.rows) { "trsm: B has ${B.rows} rows, expected ${A.rows}" }
    trsmCore(A.data, A.rows, B.data, B.cols, lower, transpose, unitDiag)
}

/** [trsv] over a flat row-major `n×n` buffer; shared with the LU and Cholesky solve internals. */
internal fun trsvCore(a: DoubleArray, n: Int, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
    if (!transpose) {
        if (lower) { // forward substitution; row runs are contiguous
            for (i in 0 until n) {
                val s = x[i] - denseDot(a, i * n, x, 0, i)
                x[i] = if (unitDiag) s else s / a[i * n + i]
            }
        } else { // back substitution
            for (i in n - 1 downTo 0) {
                val base = i * n
                val s = x[i] - denseDot(a, base + i + 1, x, i + 1, n - i - 1)
                x[i] = if (unitDiag) s else s / a[base + i]
            }
        }
    } else {
        if (lower) { // Tᵀ is upper: back substitution, column-oriented over rows of T
            for (i in n - 1 downTo 0) {
                val xi = if (unitDiag) x[i] else x[i] / a[i * n + i]
                x[i] = xi
                if (xi != 0.0) denseAxpy(x, 0, -xi, a, i * n, i)
            }
        } else { // Tᵀ is lower: forward substitution, column-oriented over rows of T
            for (i in 0 until n) {
                val base = i * n
                val xi = if (unitDiag) x[i] else x[i] / a[base + i]
                x[i] = xi
                if (xi != 0.0) denseAxpy(x, i + 1, -xi, a, base + i + 1, n - i - 1)
            }
        }
    }
}

/** [trsm] over flat row-major buffers (`a`: `n×n`, `b`: `n×nrhs`); every update runs along a
 *  contiguous row of `b`. */
@Suppress("CyclomaticComplexMethod", "LongParameterList")
internal fun trsmCore(
    a: DoubleArray,
    n: Int,
    b: DoubleArray,
    nrhs: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    if (!transpose) {
        if (lower) {
            for (i in 0 until n) {
                val base = i * n
                for (j in 0 until i) {
                    val f = a[base + j]
                    if (f != 0.0) denseAxpy(b, i * nrhs, -f, b, j * nrhs, nrhs)
                }
                if (!unitDiag) denseScale(b, i * nrhs, 1.0 / a[base + i], nrhs)
            }
        } else {
            for (i in n - 1 downTo 0) {
                val base = i * n
                for (j in i + 1 until n) {
                    val f = a[base + j]
                    if (f != 0.0) denseAxpy(b, i * nrhs, -f, b, j * nrhs, nrhs)
                }
                if (!unitDiag) denseScale(b, i * nrhs, 1.0 / a[base + i], nrhs)
            }
        }
    } else {
        if (lower) { // Tᵀ upper: finalize row i, then push its contribution up along T's row i
            for (i in n - 1 downTo 0) {
                val base = i * n
                if (!unitDiag) denseScale(b, i * nrhs, 1.0 / a[base + i], nrhs)
                for (j in 0 until i) {
                    val f = a[base + j]
                    if (f != 0.0) denseAxpy(b, j * nrhs, -f, b, i * nrhs, nrhs)
                }
            }
        } else { // Tᵀ lower: finalize row i, then push its contribution down along T's row i
            for (i in 0 until n) {
                val base = i * n
                if (!unitDiag) denseScale(b, i * nrhs, 1.0 / a[base + i], nrhs)
                for (j in i + 1 until n) {
                    val f = a[base + j]
                    if (f != 0.0) denseAxpy(b, j * nrhs, -f, b, i * nrhs, nrhs)
                }
            }
        }
    }
}
