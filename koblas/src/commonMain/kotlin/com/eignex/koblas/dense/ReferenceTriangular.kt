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
internal inline fun forEachRow(n: Int, b: F64DenseMatrix, columnOffset: Int = 0, op: (DoubleArray) -> Unit) {
    if (n == 0) return
    val rows = b.rows
    val bd = b.data
    val row = DoubleArray(n)
    for (i in 0 until rows) {
        for (j in 0 until n) row[j] = bd[i + (columnOffset + j) * rows]
        op(row)
        for (j in 0 until n) bd[i + (columnOffset + j) * rows] = row[j]
    }
}

/** Adds only runs whose source coefficients are nonzero, retaining vector kernels for every nonzero run. */
private fun axpySkippingZeroSource(
    k: F64Kernels,
    y: DoubleArray,
    yOff: Int,
    alpha: Double,
    x: DoubleArray,
    xOff: Int,
    len: Int,
) {
    var at = 0
    while (at < len) {
        while (at < len && x[xOff + at] == 0.0) at++
        val start = at
        while (at < len && x[xOff + at] != 0.0) at++
        if (at > start) axpyArithmetic(k, y, yOff + start, alpha, x, xOff + start, at - start)
    }
}

/** Dots only runs whose matrix coefficients are nonzero, retaining vector kernels for every nonzero run. */
private fun dotSkippingZeroMatrix(
    k: F64Kernels,
    a: DoubleArray,
    aOff: Int,
    x: DoubleArray,
    xOff: Int,
    len: Int,
): Double {
    var sum = 0.0
    var at = 0
    while (at < len) {
        while (at < len && a[aOff + at] == 0.0) at++
        val start = at
        while (at < len && a[aOff + at] != 0.0) at++
        if (at > start) sum += k.dot(a, aOff + start, x, xOff + start, at - start)
    }
    return sum
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
    guardZeroInput: Boolean = true,
    guardZeroMatrix: Boolean = false,
) {
    if (!transpose) {
        if (lower) { // column j contributes to x(j..), so descending keeps x(j) original
            for (j in n - 1 downTo 0) {
                val base = aOff + j + j * lda
                val xj = x[xOff + j]
                if (!guardZeroInput || xj != 0.0) {
                    x[xOff + j] = if (unitDiag) xj else a[base] * xj
                    if (guardZeroMatrix) {
                        axpySkippingZeroSource(k, x, xOff + j + 1, xj, a, base + 1, n - j - 1)
                    } else {
                        axpyArithmetic(k, x, xOff + j + 1, xj, a, base + 1, n - j - 1)
                    }
                }
            }
        } else { // column j contributes to x(0..j), so ascending keeps x(j) original
            for (j in 0 until n) {
                val xj = x[xOff + j]
                if (!guardZeroInput || xj != 0.0) {
                    x[xOff + j] = if (unitDiag) xj else a[aOff + j + j * lda] * xj
                    if (guardZeroMatrix) {
                        axpySkippingZeroSource(k, x, xOff, xj, a, aOff + j * lda, j)
                    } else {
                        axpyArithmetic(k, x, xOff, xj, a, aOff + j * lda, j)
                    }
                }
            }
        }
    } else {
        if (lower) { // Tᵀ is upper: row i of Tᵀ is column i of T, read forward from the diagonal
            for (i in 0 until n) {
                val base = aOff + i + i * lda
                val diag = if (unitDiag) x[xOff + i] else a[base] * x[xOff + i]
                val offDiagonal = if (guardZeroMatrix) {
                    dotSkippingZeroMatrix(k, a, base + 1, x, xOff + i + 1, n - i - 1)
                } else {
                    k.dot(a, base + 1, x, xOff + i + 1, n - i - 1)
                }
                x[xOff + i] = diag + offDiagonal
            }
        } else { // Tᵀ is lower: row i of Tᵀ is column i of T, read up to the diagonal
            for (i in n - 1 downTo 0) {
                val diag = if (unitDiag) x[xOff + i] else a[aOff + i + i * lda] * x[xOff + i]
                val offDiagonal = if (guardZeroMatrix) {
                    dotSkippingZeroMatrix(k, a, aOff + i * lda, x, xOff, i)
                } else {
                    k.dot(a, aOff + i * lda, x, xOff, i)
                }
                x[xOff + i] = diag + offDiagonal
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
    guardZeroPivot: Boolean = true,
    guardZeroMatrix: Boolean = false,
): Long {
    var zeroAfterNonzero = 0L
    if (!transpose) {
        if (lower) { // forward substitution, finalizing x(j) then pushing it down column j
            for (j in 0 until n) {
                val base = aOff + j + j * lda
                if (guardZeroPivot && x[xOff + j] == 0.0) continue
                val xj = if (unitDiag) x[xOff + j] else x[xOff + j] / a[base]
                x[xOff + j] = xj
                if (guardZeroPivot && xj == 0.0) zeroAfterNonzero = zeroAfterNonzero or (1L shl j)
                if (guardZeroMatrix) {
                    axpySkippingZeroSource(k, x, xOff + j + 1, -xj, a, base + 1, n - j - 1)
                } else {
                    axpyArithmetic(k, x, xOff + j + 1, -xj, a, base + 1, n - j - 1)
                }
            }
        } else { // back substitution, finalizing x(j) then pushing it up column j
            for (j in n - 1 downTo 0) {
                if (guardZeroPivot && x[xOff + j] == 0.0) continue
                val xj = if (unitDiag) x[xOff + j] else x[xOff + j] / a[aOff + j + j * lda]
                x[xOff + j] = xj
                if (guardZeroPivot && xj == 0.0) zeroAfterNonzero = zeroAfterNonzero or (1L shl j)
                if (guardZeroMatrix) {
                    axpySkippingZeroSource(k, x, xOff, -xj, a, aOff + j * lda, j)
                } else {
                    axpyArithmetic(k, x, xOff, -xj, a, aOff + j * lda, j)
                }
            }
        }
    } else {
        if (lower) { // Tᵀ is upper: back substitution, dotting column i of T behind the frontier
            for (i in n - 1 downTo 0) {
                val base = aOff + i + i * lda
                val product = if (guardZeroMatrix) {
                    dotSkippingZeroMatrix(k, a, base + 1, x, xOff + i + 1, n - i - 1)
                } else {
                    k.dot(a, base + 1, x, xOff + i + 1, n - i - 1)
                }
                val s = x[xOff + i] - product
                x[xOff + i] = if (unitDiag) s else s / a[base]
            }
        } else { // Tᵀ is lower: forward substitution, dotting column i of T behind the frontier
            for (i in 0 until n) {
                val product = if (guardZeroMatrix) {
                    dotSkippingZeroMatrix(k, a, aOff + i * lda, x, xOff, i)
                } else {
                    k.dot(a, aOff + i * lda, x, xOff, i)
                }
                val s = x[xOff + i] - product
                x[xOff + i] = if (unitDiag) s else s / a[aOff + i + i * lda]
            }
        }
    }
    return zeroAfterNonzero
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
 * Matrix operations solve or multiply diagonal blocks with the vector cores and send off-diagonal updates
 * through shared blocked level-3 kernels. A transposed triangle is read in its original storage orientation.
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
    if (solve) {
        blockedTriangularSolve(k, a.data, a.rows, b, lower, transpose, unitDiag, right)
    } else {
        blockedTriangularMultiply(k, a.data, a.rows, b, lower, transpose, unitDiag, right)
    }
}

/** Blocked TRSM over a column-major triangle. */
private fun blockedTriangularSolve(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
) {
    if (right) {
        blockedRightSolve(k, triangle, n, b, lower, transpose, unitDiag)
    } else {
        blockedLeftSolve(k, triangle, n, b, lower, transpose, unitDiag)
    }
}

private fun blockedLeftSolve(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val bd = b.data
    val nrhs = b.cols
    val effectiveLower = if (transpose) !lower else lower
    if (effectiveLower) {
        var start = 0
        while (start < n) {
            val end = min(start + REFERENCE_TRIANGULAR_BLOCK, n)
            val size = end - start
            var zeroCoefficientMasks: LongArray? = null
            for (column in 0 until nrhs) {
                val mask = trsvCore(
                    k, triangle, size, bd, start + start * n, column * n + start, n,
                    lower, transpose, unitDiag,
                )
                if (mask != 0L) {
                    val masks = zeroCoefficientMasks ?: LongArray(nrhs).also { zeroCoefficientMasks = it }
                    masks[column] = mask
                }
            }
            if (end < n) {
                blockedLeftTriangularUpdate(
                    k, -1.0, triangle, n, transpose, bd, end, n - end, start, size, nrhs,
                    zeroCoefficientMasks,
                )
            }
            start = end
        }
    } else {
        var end = n
        while (end > 0) {
            val start = max(0, end - REFERENCE_TRIANGULAR_BLOCK)
            val size = end - start
            var zeroCoefficientMasks: LongArray? = null
            for (column in 0 until nrhs) {
                val mask = trsvCore(
                    k, triangle, size, bd, start + start * n, column * n + start, n,
                    lower, transpose, unitDiag,
                )
                if (mask != 0L) {
                    val masks = zeroCoefficientMasks ?: LongArray(nrhs).also { zeroCoefficientMasks = it }
                    masks[column] = mask
                }
            }
            if (start > 0) {
                blockedLeftTriangularUpdate(
                    k, -1.0, triangle, n, transpose, bd, 0, start, start, size, nrhs,
                    zeroCoefficientMasks,
                )
            }
            end = start
        }
    }
}

@Suppress("LongParameterList")
private fun blockedLeftTriangularUpdate(
    k: F64Kernels,
    alpha: Double,
    triangle: DoubleArray,
    n: Int,
    transpose: Boolean,
    b: DoubleArray,
    rowStart: Int,
    rowCount: Int,
    innerStart: Int,
    innerCount: Int,
    columns: Int,
    zeroCoefficientMasks: LongArray? = null,
) {
    if (transpose) {
        blockedTransposedLeftUpdate(
            k, alpha, triangle, innerStart + rowStart * n, n,
            b, innerStart, n, b, rowStart, n, rowCount, columns, innerCount,
        )
    } else {
        blockedUpdate(
            k, alpha, triangle, rowStart + innerStart * n, n,
            b, innerStart, n, b, rowStart, n, rowCount, columns, innerCount,
            zeroCoefficientMasks = zeroCoefficientMasks,
        )
    }
}

private fun blockedRightSolve(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val rows = b.rows
    val bd = b.data
    val effectiveLower = if (transpose) !lower else lower
    var boundary = if (effectiveLower) n else 0
    while (if (effectiveLower) boundary > 0 else boundary < n) {
        val start = if (effectiveLower) max(0, boundary - REFERENCE_TRIANGULAR_BLOCK) else boundary
        val end = if (effectiveLower) boundary else min(boundary + REFERENCE_TRIANGULAR_BLOCK, n)
        val size = end - start
        forEachRow(size, b, start) { row ->
            trsvCore(
                k, triangle, size, row, start + start * n, 0, n, lower, !transpose, unitDiag,
                guardZeroPivot = false,
                guardZeroMatrix = true,
            )
        }
        if (effectiveLower && start > 0) {
            blockedRightTriangularUpdate(
                k, -1.0, bd, rows, triangle, n, transpose, start, size, 0, start,
            )
        } else if (!effectiveLower && end < n) {
            blockedRightTriangularUpdate(
                k, -1.0, bd, rows, triangle, n, transpose, start, size, end, n - end,
            )
        }
        boundary = if (effectiveLower) start else end
    }
}

/** Blocked TRMM over a column-major triangle. */
private fun blockedTriangularMultiply(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
) {
    if (right) {
        blockedRightMultiply(k, triangle, n, b, lower, transpose, unitDiag)
    } else {
        blockedLeftMultiply(k, triangle, n, b, lower, transpose, unitDiag)
    }
}

private fun blockedLeftMultiply(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val bd = b.data
    val nrhs = b.cols
    val effectiveLower = if (transpose) !lower else lower
    var boundary = if (effectiveLower) n else 0
    while (if (effectiveLower) boundary > 0 else boundary < n) {
        val start = if (effectiveLower) max(0, boundary - REFERENCE_TRIANGULAR_BLOCK) else boundary
        val end = if (effectiveLower) boundary else min(boundary + REFERENCE_TRIANGULAR_BLOCK, n)
        val size = end - start
        for (column in 0 until nrhs) {
            trmvCore(
                k, triangle, size, bd, start + start * n, column * n + start, n,
                lower, transpose, unitDiag,
            )
        }
        if (effectiveLower && start > 0) {
            blockedLeftTriangularUpdate(k, 1.0, triangle, n, transpose, bd, start, size, 0, start, nrhs)
        } else if (!effectiveLower && end < n) {
            blockedLeftTriangularUpdate(
                k, 1.0, triangle, n, transpose, bd, start, size, end, n - end, nrhs,
            )
        }
        boundary = if (effectiveLower) start else end
    }
}

private fun blockedRightMultiply(
    k: F64Kernels,
    triangle: DoubleArray,
    n: Int,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val bd = b.data
    val effectiveLower = if (transpose) !lower else lower
    var boundary = if (effectiveLower) 0 else n
    while (if (effectiveLower) boundary < n else boundary > 0) {
        val start = if (effectiveLower) boundary else max(0, boundary - REFERENCE_TRIANGULAR_BLOCK)
        val end = if (effectiveLower) min(boundary + REFERENCE_TRIANGULAR_BLOCK, n) else boundary
        val size = end - start
        forEachRow(size, b, start) { row ->
            trmvCore(
                k, triangle, size, row, start + start * n, 0, n, lower, !transpose, unitDiag,
                guardZeroInput = false,
                guardZeroMatrix = true,
            )
        }
        if (effectiveLower && end < n) {
            blockedRightTriangularUpdate(
                k, 1.0, bd, b.rows, triangle, n, transpose, end, n - end, start, size,
            )
        } else if (!effectiveLower && start > 0) {
            blockedRightTriangularUpdate(k, 1.0, bd, b.rows, triangle, n, transpose, 0, start, start, size)
        }
        boundary = if (effectiveLower) end else start
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
