@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, T, X

package com.eignex.koblas.dense

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSquare

/*
 * The portable triangular kernels, netlib dtrsv, dtrsm, dtrmv and dtrmm over a flat column-major buffer.
 * These are the semantic definition a native triangular routine is validated against, and what
 * [F64ReferenceBlas], [F64ReferenceLapack] and the host adapters' fallbacks call. `Triangular.kt` is the
 * public facade that routes through the installed context instead.
 */

/** Stages each row of [b] through a scratch vector and applies [op] to it, with the transpose flag
 *  flipped. */
internal inline fun forEachRow(n: Int, b: F64DenseMatrix, op: (DoubleArray) -> Unit) {
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
 * x(xOff until xOff + n). [lda] defaults to [n] and is the BLAS `lda` otherwise.
 */
@Suppress("LongParameterList") // the shape, the leading dimension and the three BLAS flags
internal fun trmvCore(
    k: F64VectorKernels,
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
        if (lower) { // column j contributes to x(j..), so descending keeps x(j) original
            for (j in n - 1 downTo 0) {
                val base = j + j * lda
                val xj = x[xOff + j]
                x[xOff + j] = if (unitDiag) xj else a[base] * xj
                if (xj != 0.0) k.axpy(x, xOff + j + 1, xj, a, base + 1, n - j - 1)
            }
        } else { // column j contributes to x(0..j), so ascending keeps x(j) original
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
    k: F64VectorKernels,
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
 *  x(xOff until xOff + n). The diagonal is not checked, so a singular triangle yields infinities or NaNs. */
@Suppress("LongParameterList") // the shape, the leading dimension and the three BLAS flags
internal fun trsvCore(
    k: F64VectorKernels,
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
        if (lower) { // forward substitution, finalizing x(j) then pushing it down column j
            for (j in 0 until n) {
                val base = j + j * lda
                val xj = if (unitDiag) x[xOff + j] else x[xOff + j] / a[base]
                x[xOff + j] = xj
                if (xj != 0.0) k.axpy(x, xOff + j + 1, -xj, a, base + 1, n - j - 1)
            }
        } else { // back substitution, finalizing x(j) then pushing it up column j
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

/**
 * The body [F64Blas.trsv] and [F64Blas.trmv] share. The two BLAS routines take the same arguments and differ only
 * in which core runs, which [solve] selects.
 *
 * The selection is a flag rather than a passed-in core so the call stays direct. A function reference here
 * would be an indirect call, and in the right-hand path of [triangularMatrix] it would be one per row.
 */
@Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
internal fun triangularVector(
    k: F64VectorKernels,
    a: F64DenseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
) {
    val what = if (solve) "trsv" else "trmv"
    requireSquare(a, what)
    requireShape(x.size == a.rows) { "$what: x length ${x.size} != ${a.rows}" }
    if (solve) {
        trsvCore(k, a.data, a.rows, x, lower = lower, transpose = transpose, unitDiag = unitDiag)
    } else {
        trmvCore(k, a.data, a.rows, x, lower = lower, transpose = transpose, unitDiag = unitDiag)
    }
}

/**
 * The body [F64Blas.trsm] and [F64Blas.trmm] share, with [solve] selecting the core as in [triangularVector].
 *
 * From the right the operands are the rows of [b] and the triangle is transposed, which is the same identity
 * both routines used when they were written out separately.
 */
@Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
internal fun triangularMatrix(
    k: F64VectorKernels,
    a: F64DenseMatrix,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    solve: Boolean,
) {
    val what = if (solve) "trsm" else "trmm"
    requireSquare(a, what)
    if (right) {
        requireShape(b.cols == a.rows) { "$what right: B has ${b.cols} cols, expected ${a.rows}" }
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        if (alpha != 1.0) k.scale(b.data, 0, alpha, b.data.size)
        forEachRow(a.rows, b) { row ->
            if (solve) {
                trsvCore(k, a.data, a.rows, row, lower = lower, transpose = !transpose, unitDiag = unitDiag)
            } else {
                trmvCore(k, a.data, a.rows, row, lower = lower, transpose = !transpose, unitDiag = unitDiag)
            }
        }
    } else {
        requireShape(b.rows == a.rows) { "$what: B has ${b.rows} rows, expected ${a.rows}" }
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        if (alpha != 1.0) k.scale(b.data, 0, alpha, b.data.size)
        if (solve) {
            trsmCore(k, a.data, a.rows, b.data, b.cols, lower, transpose, unitDiag)
        } else {
            trmmCore(k, a.data, a.rows, b.data, b.cols, lower, transpose, unitDiag)
        }
    }
}

/** [trsv] applied to each of the [nrhs] contiguous columns of a flat column-major `n×nrhs` [b]. */
@Suppress("LongParameterList")
internal fun trsmCore(
    k: F64VectorKernels,
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
