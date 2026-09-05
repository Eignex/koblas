@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, T, X

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.axpyArithmetic
import com.eignex.koblas.dense.borrowTransposed
import com.eignex.koblas.sparse.REFERENCE_SPARSE_RHS_WIDTH
import kotlin.math.min

/*
 * The portable sparse triangular kernels and the sparse-times-dense products.
 *
 * Their own file rather than members of the portable backend, for the reason SparseTranspose.kt gives for
 * the same move: a factorization needs them too, and reaching the seam would route the definition of a
 * routine through whichever backend happens to be registered.
 */

/** Visits dense right-hand sides in cache-sized panels. */
private inline fun forEachRhsPanel(columns: Int, action: (start: Int, width: Int) -> Unit) {
    var start = 0
    while (start < columns) {
        val width = min(REFERENCE_SPARSE_RHS_WIDTH, columns - start)
        action(start, width)
        start += width
    }
}

/** Runs [block] with the diagonal of [a] borrowed from [workspace], or null when [unitDiag] takes it as 1. */
internal inline fun withExplicitDiagonal(
    a: F64SparseMatrix,
    n: Int,
    unitDiag: Boolean,
    workspace: Workspace?,
    crossinline block: (DoubleArray?) -> Unit,
) {
    if (unitDiag) {
        block(null)
    } else {
        workspace.borrow(n) { diagonal ->
            for (j in 0 until n) diagonal[j] = a[j, j]
            block(diagonal)
        }
    }
}

/** Sparse triangular multiply over RHS panels, so values and indices are read once for several dense columns. */
internal fun trmmLeftCore(
    a: F64SparseMatrix,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    diagonal: DoubleArray?,
) {
    val n = a.rows
    val bd = b.data
    val work = DoubleArray(REFERENCE_SPARSE_RHS_WIDTH)
    val order = if (lower != transpose) n - 1 downTo 0 else 0 until n
    forEachRhsPanel(b.cols) { columnStart, width ->
        for (j in order) {
            val dj = diagonal?.get(j) ?: 1.0
            if (!transpose) {
                // A stored zero source lane is skipped entirely, as trmvCore does: it keeps a zero
                // lane out of both the diagonal write and
                // the scatter, so a NaN/Inf coefficient elsewhere in the column never reaches it.
                var active = 0
                for (rhs in 0 until width) {
                    val at = (columnStart + rhs) * n + j
                    val xj = bd[at]
                    work[rhs] = xj
                    if (xj != 0.0) {
                        active = active or (1 shl rhs)
                        bd[at] = if (unitDiag) xj else dj * xj
                    }
                }
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) {
                        for (rhs in 0 until width) {
                            if (active and (1 shl rhs) != 0) bd[(columnStart + rhs) * n + i] += v * work[rhs]
                        }
                    }
                }
            } else {
                for (rhs in 0 until width) {
                    val at = (columnStart + rhs) * n + j
                    work[rhs] = if (unitDiag) bd[at] else dj * bd[at]
                }
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) {
                        for (rhs in 0 until width) work[rhs] += v * bd[(columnStart + rhs) * n + i]
                    }
                }
                for (rhs in 0 until width) bd[(columnStart + rhs) * n + j] = work[rhs]
            }
        }
    }
}

/**
 * Right multiply over contiguous dense columns, which turns every sparse update into a Level 1 operation,
 * the same trade [trsmRightCore] makes. A row times op(T) is op(T)ᵀ times its column-shaped view, so this
 * walks the triangle exactly as [trmmLeftCore] does with the transpose flag flipped, only every scalar lane
 * of that algorithm is a whole column of [b] here instead of one right-hand side in a panel.
 */
internal fun trmmRightCore(
    kernels: F64Kernels,
    a: F64SparseMatrix,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    diagonal: DoubleArray?,
) {
    val rows = b.rows
    if (rows == 0) return
    val bd = b.data
    val n = a.rows
    val gather = !transpose
    val order = if (lower != gather) n - 1 downTo 0 else 0 until n
    for (l in order) {
        val lOff = l * rows
        if (gather) {
            if (!unitDiag) kernels.scale(bd, lOff, diagonal!![l], rows)
            a.forEachInColumn(l) { i, v ->
                if (v != 0.0 && (if (lower) i > l else i < l)) {
                    kernels.axpy(bd, lOff, v, bd, i * rows, rows)
                }
            }
        } else {
            // The diagonal write must follow the scatter: it overwrites this column's own slot, which
            // the scatter below still needs to read at its pre-multiply value.
            a.forEachInColumn(l) { i, v ->
                if (v != 0.0 && (if (lower) i > l else i < l)) {
                    kernels.axpy(bd, i * rows, v, bd, lOff, rows)
                }
            }
            if (!unitDiag) kernels.scale(bd, lOff, diagonal!![l], rows)
        }
    }
}

/**
 * Sparse dtrmv. Direction preserves each source before its destination is written, so no work buffer is
 * needed for the ordinary in-place case.
 *
 * The diagonal is probed as `a[j, j]`, a binary search per column. That is `trmv`'s to pay: `trmm` walks
 * several right-hand sides against one triangle and precomputes the diagonal for itself instead.
 */
internal fun trmvCore(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
    val n = a.rows
    if (!transpose) {
        val order = if (lower) n - 1 downTo 0 else 0 until n
        for (j in order) {
            val xj = x[j]
            if (xj != 0.0) {
                x[j] = if (unitDiag) xj else a[j, j] * xj
                a.forEachInColumn(j) { i, v ->
                    if ((if (lower) i > j else i < j)) {
                        x[i] += v * xj
                    }
                }
            }
        }
    } else {
        val order = if (lower) 0 until n else n - 1 downTo 0
        for (j in order) {
            var sum = if (unitDiag) {
                x[j]
            } else {
                a[j, j] * x[j]
            }
            a.forEachInColumn(j) { i, v ->
                if ((if (lower) i > j else i < j)) {
                    sum += v * x[i]
                }
            }
            x[j] = sum
        }
    }
}

/**
 * Snapshots the coefficient array only when the in-place destination aliases this matrix's live values.
 * The column pointers and row indices are shared live rather than copied: [trmvCore] never
 * mutates them, and they are documented immutable for the life of a [F64SparseMatrix].
 */
@OptIn(UnsafeKoblasApi::class)
internal fun F64SparseMatrix.stableFor(destination: DoubleArray): F64SparseMatrix = if (values === destination) {
    F64SparseMatrix.wrap(rows, cols, colPtr, rowIdx, values.copyOf())
} else {
    this
}

/** Sparse substitution over RHS panels, so values and indices are read once for several dense columns. */
internal fun trsmLeftCore(
    a: F64SparseMatrix,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    diagonal: DoubleArray?,
    work: DoubleArray,
) {
    val n = a.rows
    val bd = b.data
    val order = if (lower != transpose) 0 until n else n - 1 downTo 0
    forEachRhsPanel(b.cols) { columnStart, width ->
        for (j in order) {
            if (!transpose) {
                val divisor = diagonal?.get(j) ?: 1.0
                var active = 0
                for (rhs in 0 until width) {
                    val at = (columnStart + rhs) * n + j
                    val raw = bd[at]
                    work[rhs] = if (raw == 0.0) 0.0 else raw / divisor
                    if (raw != 0.0) {
                        active = active or (1 shl rhs)
                        bd[at] = work[rhs]
                    }
                }
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) {
                        for (rhs in 0 until width) {
                            val xj = work[rhs]
                            if (active and (1 shl rhs) != 0) bd[(columnStart + rhs) * n + i] -= v * xj
                        }
                    }
                }
            } else {
                for (rhs in 0 until width) work[rhs] = bd[(columnStart + rhs) * n + j]
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) {
                        for (rhs in 0 until width) work[rhs] -= v * bd[(columnStart + rhs) * n + i]
                    }
                }
                val divisor = diagonal?.get(j) ?: 1.0
                for (rhs in 0 until width) bd[(columnStart + rhs) * n + j] = work[rhs] / divisor
            }
        }
    }
}

/** Right solve over contiguous dense columns, which turns every sparse update into a Level 1 operation. */
internal fun trsmRightCore(
    kernels: F64Kernels,
    a: F64SparseMatrix,
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    diagonal: DoubleArray?,
) {
    val rows = b.rows
    if (rows == 0) return
    val bd = b.data
    val order = if (lower != transpose) 0 until a.rows else a.rows - 1 downTo 0
    for (j in order) {
        val jOff = j * rows
        if (!transpose) {
            if (diagonal != null) kernels.scale(bd, jOff, 1.0 / diagonal[j], rows)
            a.forEachInColumn(j) { i, v ->
                if (v != 0.0 && (if (lower) i > j else i < j)) {
                    kernels.axpy(bd, i * rows, -v, bd, jOff, rows)
                }
            }
        } else {
            a.forEachInColumn(j) { i, v ->
                if (v != 0.0 && (if (lower) i > j else i < j)) {
                    kernels.axpy(bd, jOff, -v, bd, i * rows, rows)
                }
            }
            if (diagonal != null) kernels.scale(bd, jOff, 1.0 / diagonal[j], rows)
        }
    }
}

/** `trsv` over the `n` entries of [x], with the triangle flags resolved once by the caller. */
@Suppress("LongParameterList") // the three BLAS triangle flags
internal fun trsvCore(
    a: F64SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    val n = a.rows
    // Forward when a finished unknown feeds later columns, backward when it feeds earlier ones.
    val order = if (lower != transpose) 0 until n else n - 1 downTo 0
    for (j in order) {
        if (!transpose) {
            val raw = x[j]
            if (raw == 0.0) continue
            val xj = if (unitDiag) raw else raw / a[j, j]
            x[j] = xj
            a.forEachInColumn(j) { i, v ->
                if (if (lower) i > j else i < j) x[i] -= v * xj
            }
        } else {
            var s = x[j]
            a.forEachInColumn(j) { i, v ->
                if (if (lower) i > j else i < j) s -= v * x[i]
            }
            x[j] = if (unitDiag) s else s / a[j, j]
        }
    }
}

@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheLeft(
    alpha: Double,
    a: F64SparseMatrix,
    transposeA: Boolean,
    b: F64DenseMatrix,
    transposeB: Boolean,
    c: F64DenseMatrix,
    m: Int,
    n: Int,
    k: Int,
    workspace: Workspace?,
) {
    val cd = c.data
    val bd = b.data
    val ld = b.rows
    workspace.borrow(REFERENCE_SPARSE_RHS_WIDTH) { work ->
        forEachRhsPanel(n) { columnStart, width ->
            if (transposeA) {
                for (i in 0 until m) {
                    work.fill(0.0, 0, width)
                    a.forEachInColumn(i) { j, v ->
                        for (rhs in 0 until width) {
                            val l = columnStart + rhs
                            work[rhs] += v * bd[if (transposeB) l + j * ld else j + l * ld]
                        }
                    }
                    for (rhs in 0 until width) cd[(columnStart + rhs) * m + i] += alpha * work[rhs]
                }
            } else {
                for (j in 0 until k) {
                    for (rhs in 0 until width) {
                        val l = columnStart + rhs
                        val raw = bd[if (transposeB) l + j * ld else j + l * ld]
                        work[rhs] = alpha * raw
                    }
                    a.forEachInColumn(j) { i, v ->
                        for (rhs in 0 until width) {
                            cd[(columnStart + rhs) * m + i] += v * work[rhs]
                        }
                    }
                }
            }
        }
    }
}

/**
 * `C += alpha · op(B) · op(A)`. Each stored entry of the sparse operand scales one column of the dense
 * one into one column of the destination, so the walk is over storage either way the sparse operand is
 * transposed and only which index names which column changes.
 */
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheRight(
    kernels: F64Kernels,
    alpha: Double,
    a: F64SparseMatrix,
    transposeA: Boolean,
    b: F64DenseMatrix,
    transposeB: Boolean,
    c: F64DenseMatrix,
    m: Int,
    workspace: Workspace?,
) {
    val cd = c.data
    if (transposeB) {
        workspace.borrowTransposed(b.data, b.rows, b.cols) { packed ->
            multiplyFromTheRightColumns(kernels, alpha, a, transposeA, cd, packed, m, m)
        }
    } else {
        multiplyFromTheRightColumns(kernels, alpha, a, transposeA, cd, b.data, m, b.rows)
    }
}

internal fun multiplyFromTheRightColumns(
    kernels: F64Kernels,
    alpha: Double,
    a: F64SparseMatrix,
    transposeA: Boolean,
    c: DoubleArray,
    b: DoubleArray,
    rows: Int,
    leadingDimension: Int,
) {
    for (column in 0 until a.cols) {
        a.forEachInColumn(column) { row, v ->
            val cOff = (if (transposeA) row else column) * rows
            val bColumn = if (transposeA) column else row
            axpyArithmetic(kernels, c, cOff, alpha * v, b, bColumn * leadingDimension, rows)
        }
    }
}
