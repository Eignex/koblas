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
 * Solve `op(T) · X = B` in place, or `X · op(T) = B` when [right] (BLAS `dtrsm`): [B] holds the
 * right-hand sides on entry and the solutions on return. `T` is the [lower] or upper triangle of the
 * square [A]; `op` transposes when [transpose]; with [unitDiag] the diagonal is taken as 1 and never
 * read. From the left the right-hand sides are the columns of [B]; from the right, its rows.
 */
@Suppress("LongParameterList") // the BLAS dtrsm flag set
fun trsm(
    A: DenseMatrix,
    B: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
) {
    require(A.rows == A.cols) { "trsm requires a square matrix; got ${A.rows}x${A.cols}" }
    if (right) {
        require(B.cols == A.rows) { "trsm right: B has ${B.cols} cols, expected ${A.rows}" }
        // Row i of X satisfies op(T)ᵀ · X[i,:]ᵀ = B[i,:]ᵀ, a plain trsv with the transpose flipped;
        // rows are contiguous in the row-major backing, staged through a scratch row.
        val n = A.rows
        if (n == 0) return
        val row = DoubleArray(n)
        for (i in 0 until B.rows) {
            B.data.copyInto(row, 0, i * n, (i + 1) * n)
            trsvCore(A.data, n, row, lower, !transpose, unitDiag)
            row.copyInto(B.data, i * n)
        }
    } else {
        require(B.rows == A.rows) { "trsm: B has ${B.rows} rows, expected ${A.rows}" }
        trsmCore(A.data, A.rows, B.data, B.cols, lower, transpose, unitDiag)
    }
}

/**
 * Multiply `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. `T` is the
 * [lower] or upper triangle of the square [A]; `op` transposes when [transpose]; with [unitDiag] the
 * diagonal is taken as 1 and never read. Only the selected triangle is read.
 */
fun trmv(A: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) {
    require(A.rows == A.cols) { "trmv requires a square matrix; got ${A.rows}x${A.cols}" }
    require(x.size == A.rows) { "trmv: x length ${x.size} != ${A.rows}" }
    trmvCore(A.data, A.rows, x, lower, transpose, unitDiag)
}

/**
 * Multiply `B = op(T) · B` in place, or `B = B · op(T)` when [right] (BLAS `dtrmm`), the product
 * counterpart of [trsm]. `T` is the [lower] or upper triangle of the square [A]; `op` transposes
 * when [transpose]; with [unitDiag] the diagonal is taken as 1 and never read. Only the selected
 * triangle is read.
 */
@Suppress("LongParameterList") // the BLAS dtrmm flag set
fun trmm(
    A: DenseMatrix,
    B: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
) {
    require(A.rows == A.cols) { "trmm requires a square matrix; got ${A.rows}x${A.cols}" }
    if (right) {
        require(B.cols == A.rows) { "trmm right: B has ${B.cols} cols, expected ${A.rows}" }
        // Row i of the product is (op(T)ᵀ · B[i,:]ᵀ)ᵀ, a plain trmv with the transpose flipped.
        val n = A.rows
        if (n == 0) return
        val row = DoubleArray(n)
        for (i in 0 until B.rows) {
            B.data.copyInto(row, 0, i * n, (i + 1) * n)
            trmvCore(A.data, n, row, lower, !transpose, unitDiag)
            row.copyInto(B.data, i * n)
        }
    } else {
        require(B.rows == A.rows) { "trmm: B has ${B.rows} rows, expected ${A.rows}" }
        trmmCore(A.data, A.rows, B.data, B.cols, lower, transpose, unitDiag)
    }
}

/** [trmv] over a flat row-major `n×n` buffer. Traversal order keeps every read of `x` an original
 *  value: the non-transposed directions consume finalized entries only behind the frontier, the
 *  transposed directions push a row's contribution before its own entry is overwritten. */
internal fun trmvCore(a: DoubleArray, n: Int, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
    if (!transpose) {
        if (lower) { // x_i = Σ_{j≤i} T[i,j] x_j; descending keeps x[0..i-1] original
            for (i in n - 1 downTo 0) {
                val base = i * n
                val diag = if (unitDiag) x[i] else a[base + i] * x[i]
                x[i] = diag + denseDot(a, base, x, 0, i)
            }
        } else { // x_i = Σ_{j≥i} T[i,j] x_j; ascending keeps x[i+1..] original
            for (i in 0 until n) {
                val base = i * n
                val diag = if (unitDiag) x[i] else a[base + i] * x[i]
                x[i] = diag + denseDot(a, base + i + 1, x, i + 1, n - i - 1)
            }
        }
    } else {
        if (lower) { // Tᵀ is upper: push row i's contribution into x[0..i-1] while x[i] is original
            for (i in 0 until n) {
                val base = i * n
                val xi = x[i]
                x[i] = if (unitDiag) xi else a[base + i] * xi
                if (xi != 0.0) denseAxpy(x, 0, xi, a, base, i)
            }
        } else { // Tᵀ is lower: push row i's contribution into x[i+1..] while x[i] is original
            for (i in n - 1 downTo 0) {
                val base = i * n
                val xi = x[i]
                x[i] = if (unitDiag) xi else a[base + i] * xi
                if (xi != 0.0) denseAxpy(x, i + 1, xi, a, base + i + 1, n - i - 1)
            }
        }
    }
}

/** [trmm] over flat row-major buffers (`a`: `n×n`, `b`: `n×nrhs`); every update runs along a
 *  contiguous row of `b`, with the same original-value traversal orders as [trmvCore]. */
@Suppress("CyclomaticComplexMethod", "LongParameterList")
internal fun trmmCore(
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
            for (i in n - 1 downTo 0) {
                val base = i * n
                if (!unitDiag) denseScale(b, i * nrhs, a[base + i], nrhs)
                for (j in 0 until i) {
                    val f = a[base + j]
                    if (f != 0.0) denseAxpy(b, i * nrhs, f, b, j * nrhs, nrhs)
                }
            }
        } else {
            for (i in 0 until n) {
                val base = i * n
                if (!unitDiag) denseScale(b, i * nrhs, a[base + i], nrhs)
                for (j in i + 1 until n) {
                    val f = a[base + j]
                    if (f != 0.0) denseAxpy(b, i * nrhs, f, b, j * nrhs, nrhs)
                }
            }
        }
    } else {
        if (lower) { // Tᵀ upper: push row i upward, then apply its diagonal
            for (i in 0 until n) {
                val base = i * n
                for (j in 0 until i) {
                    val f = a[base + j]
                    if (f != 0.0) denseAxpy(b, j * nrhs, f, b, i * nrhs, nrhs)
                }
                if (!unitDiag) denseScale(b, i * nrhs, a[base + i], nrhs)
            }
        } else { // Tᵀ lower: push row i downward, then apply its diagonal
            for (i in n - 1 downTo 0) {
                val base = i * n
                for (j in i + 1 until n) {
                    val f = a[base + j]
                    if (f != 0.0) denseAxpy(b, j * nrhs, f, b, i * nrhs, nrhs)
                }
                if (!unitDiag) denseScale(b, i * nrhs, a[base + i], nrhs)
            }
        }
    }
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
