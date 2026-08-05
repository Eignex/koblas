@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, T, X

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.koblas

// Triangular solves (BLAS dtrsv / dtrsm) over the column-major flat backing of [DenseMatrix].
//
// Only the triangle selected by `lower` is ever read — the opposite triangle may hold anything, so
// the packed LU buffer (unit-lower L below the diagonal, U on and above) and a Cholesky factor both
// work as inputs directly. Column `j` of the stored triangle is a contiguous run: rows `j..n-1` for a
// lower triangle, rows `0..j` for an upper one.
//
// The non-transposed directions are column-oriented — once an unknown is final, its contribution is
// subtracted from the remaining right-hand side along a contiguous column via [VectorKernels.axpy]. The
// transposed directions read a row of `op(T)`, which is a column of `T`, so they reduce to [VectorKernels.dot].
// Both shapes are contiguous, which is why the matrix forms below are just a loop over the columns of
// `B` rather than a separate kernel.
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
 * transpose flag flipped.
 *
 * Rows are strided in this layout, so staging is a gather and a scatter rather than two bulk copies.
 * That is the cost of reusing one kernel for a path that is otherwise four more branches; the right-side
 * forms are the least-used ones in the library, and both host backends dispatch them natively above
 * their gates.
 */
internal inline fun forEachRow(n: Int, b: DenseMatrix, op: (DoubleArray) -> Unit) {
    if (n == 0) return
    val rows = b.rows
    val bd = b.data
    val row = DoubleArray(n)
    for (i in 0 until rows) {
        for (j in 0 until n) row[j] = bd[i + j * rows]
        op(row)
        for (j in 0 until n) bd[i + j * rows] = row[j]
    }
}

/**
 * [trmv] over the leading `n×n` triangle of a column-major [a] whose columns are [lda] apart, applied to
 * `x[xOff until xOff + n]`.
 *
 * [lda] defaults to [n] for a buffer that holds nothing but the triangle, and is the BLAS `lda`
 * otherwise: the `R` of an `m×n` QR sits in the leading `n` rows of an `m`-strided buffer, which under
 * row-major storage happened to be contiguous and here is not.
 *
 * Traversal order keeps every read of `x` an original value: the non-transposed directions push a
 * column's contribution before its own entry is overwritten, the transposed directions consume finalized
 * entries only behind the frontier.
 */
@Suppress("LongParameterList") // the shape, the leading dimension and the three BLAS flags
internal fun trmvCore(
    k: VectorKernels,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    xOff: Int = 0,
    lda: Int = n,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    if (!transpose) {
        if (lower) { // column j contributes to x[j..]; descending keeps x[j] original
            for (j in n - 1 downTo 0) {
                val base = j + j * lda
                val xj = x[xOff + j]
                x[xOff + j] = if (unitDiag) xj else a[base] * xj
                if (xj != 0.0) k.axpy(x, xOff + j + 1, xj, a, base + 1, n - j - 1)
            }
        } else { // column j contributes to x[0..j]; ascending keeps x[j] original
            for (j in 0 until n) {
                val xj = x[xOff + j]
                x[xOff + j] = if (unitDiag) xj else a[j + j * lda] * xj
                if (xj != 0.0) k.axpy(x, xOff, xj, a, j * lda, j)
            }
        }
    } else {
        if (lower) { // Tᵀ is upper: row i of Tᵀ is column i of T, read forward from the diagonal
            for (i in 0 until n) {
                val base = i + i * lda
                val diag = if (unitDiag) x[xOff + i] else a[base] * x[xOff + i]
                x[xOff + i] = diag + k.dot(a, base + 1, x, xOff + i + 1, n - i - 1)
            }
        } else { // Tᵀ is lower: row i of Tᵀ is column i of T, read up to the diagonal
            for (i in n - 1 downTo 0) {
                val diag = if (unitDiag) x[xOff + i] else a[i + i * lda] * x[xOff + i]
                x[xOff + i] = diag + k.dot(a, i * lda, x, xOff, i)
            }
        }
    }
}

/** [trmv] applied to each of the [nrhs] contiguous columns of a flat column-major `n×nrhs` [b]. */
@Suppress("LongParameterList")
internal fun trmmCore(
    k: VectorKernels,
    a: DoubleArray,
    n: Int,
    b: DoubleArray,
    nrhs: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    for (c in 0 until nrhs) {
        trmvCore(k, a, n, b, c * n, lower = lower, transpose = transpose, unitDiag = unitDiag)
    }
}

/** [trsv] over the leading `n×n` triangle of a column-major [a] with leading dimension [lda], applied to
 *  `x[xOff until xOff + n]`; shared with the LU, QR and Cholesky solve internals. See [trmvCore] for
 *  what [lda] is for. */
@Suppress("LongParameterList") // the shape, the leading dimension and the three BLAS flags
internal fun trsvCore(
    k: VectorKernels,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    xOff: Int = 0,
    lda: Int = n,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    if (!transpose) {
        if (lower) { // forward substitution: finalize x[j], then push it down column j
            for (j in 0 until n) {
                val base = j + j * lda
                val xj = if (unitDiag) x[xOff + j] else x[xOff + j] / a[base]
                x[xOff + j] = xj
                if (xj != 0.0) k.axpy(x, xOff + j + 1, -xj, a, base + 1, n - j - 1)
            }
        } else { // back substitution: finalize x[j], then push it up column j
            for (j in n - 1 downTo 0) {
                val xj = if (unitDiag) x[xOff + j] else x[xOff + j] / a[j + j * lda]
                x[xOff + j] = xj
                if (xj != 0.0) k.axpy(x, xOff, -xj, a, j * lda, j)
            }
        }
    } else {
        if (lower) { // Tᵀ is upper: back substitution, dotting column i of T behind the frontier
            for (i in n - 1 downTo 0) {
                val base = i + i * lda
                val s = x[xOff + i] - k.dot(a, base + 1, x, xOff + i + 1, n - i - 1)
                x[xOff + i] = if (unitDiag) s else s / a[base]
            }
        } else { // Tᵀ is lower: forward substitution, dotting column i of T behind the frontier
            for (i in 0 until n) {
                val s = x[xOff + i] - k.dot(a, i * lda, x, xOff, i)
                x[xOff + i] = if (unitDiag) s else s / a[i + i * lda]
            }
        }
    }
}

/** [trsv] applied to each of the [nrhs] contiguous columns of a flat column-major `n×nrhs` [b]. */
@Suppress("LongParameterList")
internal fun trsmCore(
    k: VectorKernels,
    a: DoubleArray,
    n: Int,
    b: DoubleArray,
    nrhs: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    for (c in 0 until nrhs) {
        trsvCore(k, a, n, b, c * n, lower = lower, transpose = transpose, unitDiag = unitDiag)
    }
}
