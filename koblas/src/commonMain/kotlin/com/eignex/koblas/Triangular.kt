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

// The triangular routines below dispatch to the installed backend, which is what makes them fast on a
// target with a host BLAS. Each is the free-function spelling of the LinearAlgebra member of the same
// name; call either. In all of them T is the lower or upper triangle of the square first argument, op
// transposes when `transpose`, and with `unitDiag` the diagonal is taken as 1 and never read, so the
// rest of the matrix may hold anything.

/** Solve `op(T) · x = b` in place (BLAS `dtrsv`); see [LinearAlgebra.trsv]. */
fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) =
    koblas.trsv(a, x, lower, transpose, unitDiag)

/** Solve `op(T) · X = B`, or `X · op(T) = B` when [right] (BLAS `dtrsm`); see [LinearAlgebra.trsm]. */
@Suppress("LongParameterList") // the BLAS dtrsm signature
fun trsm(
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
) = koblas.trsm(a, b, lower, transpose, unitDiag, right)

/** Multiply `x = op(T) · x` in place (BLAS `dtrmv`); see [LinearAlgebra.trmv]. */
fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) =
    koblas.trmv(a, x, lower, transpose, unitDiag)

/** Multiply `B = op(T) · B`, or `B = B · op(T)` when [right] (BLAS `dtrmm`); see [LinearAlgebra.trmm]. */
@Suppress("LongParameterList") // the BLAS dtrmm signature
fun trmm(
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
) = koblas.trmm(a, b, lower, transpose, unitDiag, right)

/**
 * Stages each row of [b] through a scratch vector and applies [op] to it, which is how the right-side
 * forms reuse the vector kernels: row `i` of the result is `(op(T)ᵀ · B[i,:]ᵀ)ᵀ`, a vector call with the
 * transpose flag flipped. Rows are contiguous in the row-major backing, so staging is two copies.
 */
internal inline fun forEachRow(n: Int, b: DenseMatrix, op: (DoubleArray) -> Unit) {
    if (n == 0) return
    val row = DoubleArray(n)
    for (i in 0 until b.rows) {
        b.data.copyInto(row, 0, i * n, (i + 1) * n)
        op(row)
        row.copyInto(b.data, i * n)
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
