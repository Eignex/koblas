@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, T, X

package com.eignex.koblas.dense

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSquare
import kotlin.math.max
import kotlin.math.min

/*
 * The portable triangular kernels, netlib dtrsv, dtrsm, dtrmv and dtrmm over a flat column-major buffer.
 * These are the semantic definition a native triangular routine is validated against, and what
 * [F64ReferenceBlas], [F64ReferenceDecompositions] and the host adapters' fallbacks call. `Triangular.kt` is the
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
    k: F64Kernels,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    aOff: Int = 0,
    xOff: Int = 0,
    lda: Int = n,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    if (!transpose) {
        if (lower) { // column j contributes to x(j..), so descending keeps x(j) original
            for (j in n - 1 downTo 0) {
                val base = aOff + j + j * lda
                val xj = x[xOff + j]
                x[xOff + j] = if (unitDiag) xj else a[base] * xj
                if (xj != 0.0) k.axpy(x, xOff + j + 1, xj, a, base + 1, n - j - 1)
            }
        } else { // column j contributes to x(0..j), so ascending keeps x(j) original
            for (j in 0 until n) {
                val xj = x[xOff + j]
                x[xOff + j] = if (unitDiag) xj else a[aOff + j + j * lda] * xj
                if (xj != 0.0) k.axpy(x, xOff, xj, a, aOff + j * lda, j)
            }
        }
    } else {
        if (lower) { // Tᵀ is upper: row i of Tᵀ is column i of T, read forward from the diagonal
            for (i in 0 until n) {
                val base = aOff + i + i * lda
                val diag = if (unitDiag) x[xOff + i] else a[base] * x[xOff + i]
                x[xOff + i] = diag + k.dot(a, base + 1, x, xOff + i + 1, n - i - 1)
            }
        } else { // Tᵀ is lower: row i of Tᵀ is column i of T, read up to the diagonal
            for (i in n - 1 downTo 0) {
                val diag = if (unitDiag) x[xOff + i] else a[aOff + i + i * lda] * x[xOff + i]
                x[xOff + i] = diag + k.dot(a, aOff + i * lda, x, xOff, i)
            }
        }
    }
}

/** [trsv] over the leading `n×n` triangle of a column-major [a] with leading dimension [lda], applied to
 *  x(xOff until xOff + n). The diagonal is not checked, so a singular triangle yields infinities or NaNs. */
@Suppress("LongParameterList") // the shape, the leading dimension and the three BLAS flags
internal fun trsvCore(
    k: F64Kernels,
    a: DoubleArray,
    n: Int,
    x: DoubleArray,
    aOff: Int = 0,
    xOff: Int = 0,
    lda: Int = n,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    if (!transpose) {
        if (lower) { // forward substitution, finalizing x(j) then pushing it down column j
            for (j in 0 until n) {
                val base = aOff + j + j * lda
                val xj = if (unitDiag) x[xOff + j] else x[xOff + j] / a[base]
                x[xOff + j] = xj
                if (xj != 0.0) k.axpy(x, xOff + j + 1, -xj, a, base + 1, n - j - 1)
            }
        } else { // back substitution, finalizing x(j) then pushing it up column j
            for (j in n - 1 downTo 0) {
                val xj = if (unitDiag) x[xOff + j] else x[xOff + j] / a[aOff + j + j * lda]
                x[xOff + j] = xj
                if (xj != 0.0) k.axpy(x, xOff, -xj, a, aOff + j * lda, j)
            }
        }
    } else {
        if (lower) { // Tᵀ is upper: back substitution, dotting column i of T behind the frontier
            for (i in n - 1 downTo 0) {
                val base = aOff + i + i * lda
                val s = x[xOff + i] - k.dot(a, base + 1, x, xOff + i + 1, n - i - 1)
                x[xOff + i] = if (unitDiag) s else s / a[base]
            }
        } else { // Tᵀ is lower: forward substitution, dotting column i of T behind the frontier
            for (i in 0 until n) {
                val s = x[xOff + i] - k.dot(a, aOff + i * lda, x, xOff, i)
                x[xOff + i] = if (unitDiag) s else s / a[aOff + i + i * lda]
            }
        }
    }
}

/**
 * The body [F64Blas.trsv] and [F64Blas.trmv] share. The two BLAS routines take the same arguments and differ only
 * in which core runs, which [solve] selects.
 *
 * The selection is a flag rather than a passed-in core so the call stays direct.
 */
@Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
internal fun triangularVector(
    k: F64Kernels,
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
 * Matrix operations pack a transposed triangle once, solve or multiply diagonal blocks with the vector
 * cores, and send off-diagonal updates through the shared blocked level-3 kernel.
 */
@Suppress("LongParameterList") // the shared BLAS signature plus the entry-point flag
internal fun triangularMatrix(
    k: F64Kernels,
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
    } else {
        requireShape(b.rows == a.rows) { "$what: B has ${b.rows} rows, expected ${a.rows}" }
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        if (alpha != 1.0) k.scale(b.data, 0, alpha, b.data.size)
    }
    val packed = if (transpose) packTriangularOp(a.data, a.rows, lower, true, unitDiag) else a.data
    val effectiveLower = if (transpose) !lower else lower
    val packedUnitDiag = unitDiag && !transpose
    if (solve) {
        blockedTriangularSolve(k, packed, a.rows, b, effectiveLower, packedUnitDiag, right)
    } else {
        blockedTriangularMultiply(k, packed, a.rows, b, effectiveLower, packedUnitDiag, right)
    }
}

/** Blocked TRSM over a column-major `op(T)`, packed when the public operation requested a transpose. */
private fun blockedTriangularSolve(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    unitDiag: Boolean,
    right: Boolean,
) {
    if (right) {
        blockedRightSolve(k, triangle, n, b, lower, unitDiag)
    } else {
        blockedLeftSolve(k, triangle, n, b, lower, unitDiag)
    }
}

private fun blockedLeftSolve(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    unitDiag: Boolean,
) {
    val bd = b.data
    val nrhs = b.cols
    if (lower) {
        var start = 0
        while (start < n) {
            val end = min(start + REFERENCE_TRIANGULAR_BLOCK, n)
            val size = end - start
            for (column in 0 until nrhs) {
                trsvCore(k, triangle, size, bd, start + start * n, column * n + start, n, true, false, unitDiag)
            }
            if (end < n) {
                blockedUpdate(k, -1.0, triangle, end + start * n, n, bd, start, n, bd, end, n, n - end, nrhs, size)
            }
            start = end
        }
    } else {
        var end = n
        while (end > 0) {
            val start = max(0, end - REFERENCE_TRIANGULAR_BLOCK)
            val size = end - start
            for (column in 0 until nrhs) {
                trsvCore(k, triangle, size, bd, start + start * n, column * n + start, n, false, false, unitDiag)
            }
            if (start > 0) {
                blockedUpdate(k, -1.0, triangle, start * n, n, bd, start, n, bd, 0, n, start, nrhs, size)
            }
            end = start
        }
    }
}

private fun blockedRightSolve(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    unitDiag: Boolean,
) {
    val rows = b.rows
    val bd = b.data
    val row = DoubleArray(REFERENCE_TRIANGULAR_BLOCK)
    var boundary = if (lower) n else 0
    while (if (lower) boundary > 0 else boundary < n) {
        val start = if (lower) max(0, boundary - REFERENCE_TRIANGULAR_BLOCK) else boundary
        val end = if (lower) boundary else min(boundary + REFERENCE_TRIANGULAR_BLOCK, n)
        val size = end - start
        for (i in 0 until rows) {
            for (j in 0 until size) row[j] = bd[i + (start + j) * rows]
            trsvCore(k, triangle, size, row, start + start * n, 0, n, lower, true, unitDiag)
            for (j in 0 until size) bd[i + (start + j) * rows] = row[j]
        }
        if (lower && start > 0) {
            blockedUpdate(k, -1.0, bd, start * rows, rows, triangle, start, n, bd, 0, rows, rows, start, size)
        } else if (!lower && end < n) {
            blockedUpdate(
                k, -1.0, bd, start * rows, rows, triangle, start + end * n, n,
                bd, end * rows, rows, rows, n - end, size,
            )
        }
        boundary = if (lower) start else end
    }
}

/** Blocked TRMM over a column-major `op(T)`, packed when the public operation requested a transpose. */
private fun blockedTriangularMultiply(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    unitDiag: Boolean,
    right: Boolean,
) {
    if (right) {
        blockedRightMultiply(k, triangle, n, b, lower, unitDiag)
    } else {
        blockedLeftMultiply(k, triangle, n, b, lower, unitDiag)
    }
}

private fun blockedLeftMultiply(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    unitDiag: Boolean,
) {
    val bd = b.data
    val nrhs = b.cols
    var boundary = if (lower) n else 0
    while (if (lower) boundary > 0 else boundary < n) {
        val start = if (lower) max(0, boundary - REFERENCE_TRIANGULAR_BLOCK) else boundary
        val end = if (lower) boundary else min(boundary + REFERENCE_TRIANGULAR_BLOCK, n)
        val size = end - start
        for (column in 0 until nrhs) {
            trmvCore(k, triangle, size, bd, start + start * n, column * n + start, n, lower, false, unitDiag)
        }
        if (lower && start > 0) {
            blockedUpdate(k, 1.0, triangle, start, n, bd, 0, n, bd, start, n, size, nrhs, start)
        } else if (!lower && end < n) {
            blockedUpdate(
                k, 1.0, triangle, start + end * n, n, bd, end, n,
                bd, start, n, size, nrhs, n - end,
            )
        }
        boundary = if (lower) start else end
    }
}

private fun blockedRightMultiply(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    unitDiag: Boolean,
) {
    val rows = b.rows
    val bd = b.data
    val row = DoubleArray(REFERENCE_TRIANGULAR_BLOCK)
    var boundary = if (lower) 0 else n
    while (if (lower) boundary < n else boundary > 0) {
        val start = if (lower) boundary else max(0, boundary - REFERENCE_TRIANGULAR_BLOCK)
        val end = if (lower) min(boundary + REFERENCE_TRIANGULAR_BLOCK, n) else boundary
        val size = end - start
        for (i in 0 until rows) {
            for (j in 0 until size) row[j] = bd[i + (start + j) * rows]
            trmvCore(k, triangle, size, row, start + start * n, 0, n, lower, true, unitDiag)
            for (j in 0 until size) bd[i + (start + j) * rows] = row[j]
        }
        if (lower && end < n) {
            blockedUpdate(
                k, 1.0, bd, end * rows, rows, triangle, end + start * n, n,
                bd, start * rows, rows, rows, size, n - end,
            )
        } else if (!lower && start > 0) {
            blockedUpdate(k, 1.0, bd, 0, rows, triangle, start * n, n, bd, start * rows, rows, rows, size, start)
        }
        boundary = if (lower) end else start
    }
}

/** Applies [trsvCore] to contiguous columns held by decomposition storage that contains both triangles. */
@Suppress("LongParameterList")
internal fun trsmCore(
    k: F64Kernels,
    a: DoubleArray,
    n: Int,
    b: DoubleArray,
    nrhs: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    for (column in 0 until nrhs) {
        trsvCore(k, a, n, b, xOff = column * n, lower = lower, transpose = transpose, unitDiag = unitDiag)
    }
}
