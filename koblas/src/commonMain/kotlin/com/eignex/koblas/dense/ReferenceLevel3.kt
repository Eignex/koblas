package com.eignex.koblas.dense

import kotlin.math.min

/** Cache tiles for the portable level-3 routines. They are deliberately target-neutral starting values. */
internal const val REFERENCE_MC: Int = 256
internal const val REFERENCE_NC: Int = 8
internal const val REFERENCE_KC: Int = 128
internal const val REFERENCE_TRIANGULAR_BLOCK: Int = 64
private const val REFERENCE_TRANSPOSE_BLOCK: Int = 32

/** Transposes a column-major matrix into another column-major buffer using cache-sized square tiles. */
internal fun transposeBlocked(src: DoubleArray, rows: Int, cols: Int, dst: DoubleArray) {
    var column = 0
    while (column < cols) {
        val columnEnd = min(column + REFERENCE_TRANSPOSE_BLOCK, cols)
        var row = 0
        while (row < rows) {
            val rowEnd = min(row + REFERENCE_TRANSPOSE_BLOCK, rows)
            for (j in column until columnEnd) {
                val source = j * rows
                for (i in row until rowEnd) dst[j + i * cols] = src[source + i]
            }
            row = rowEnd
        }
        column = columnEnd
    }
}

/**
 * Adds `alpha * A * B` to C. All three operands are column-major panels with explicit offsets and leading
 * dimensions. The output dimensions are `m x n`, and the shared dimension is `depth`.
 */
@Suppress("LongParameterList")
internal fun blockedUpdate(
    kernels: F64Kernels,
    alpha: Double,
    a: DoubleArray,
    aOff: Int,
    lda: Int,
    b: DoubleArray,
    bOff: Int,
    ldb: Int,
    c: DoubleArray,
    cOff: Int,
    ldc: Int,
    m: Int,
    n: Int,
    depth: Int,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + REFERENCE_NC, n)
        var row = 0
        while (row < m) {
            val rowEnd = min(row + REFERENCE_MC, m)
            val length = rowEnd - row
            var inner = 0
            while (inner < depth) {
                val innerEnd = min(inner + REFERENCE_KC, depth)
                for (p in inner until innerEnd) {
                    val source = aOff + row + p * lda
                    for (j in column until columnEnd) {
                        val multiplier = alpha * b[bOff + p + j * ldb]
                        if (multiplier != 0.0) {
                            kernels.axpy(c, cOff + row + j * ldc, multiplier, a, source, length)
                        }
                    }
                }
                inner = innerEnd
            }
            row = rowEnd
        }
        column = columnEnd
    }
}

/** GEMM's blocked update, accepting a transposed B without materialising it. A is always stored `m x depth`. */
@Suppress("LongParameterList")
internal fun blockedGemmUpdate(
    kernels: F64Kernels,
    alpha: Double,
    a: DoubleArray,
    b: DoubleArray,
    bRows: Int,
    transposeB: Boolean,
    c: DoubleArray,
    m: Int,
    n: Int,
    depth: Int,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + REFERENCE_NC, n)
        var row = 0
        while (row < m) {
            val rowEnd = min(row + REFERENCE_MC, m)
            val length = rowEnd - row
            var inner = 0
            while (inner < depth) {
                val innerEnd = min(inner + REFERENCE_KC, depth)
                for (p in inner until innerEnd) {
                    val source = row + p * m
                    for (j in column until columnEnd) {
                        val bv = if (transposeB) b[j + p * bRows] else b[p + j * bRows]
                        val multiplier = alpha * bv
                        if (multiplier != 0.0) kernels.axpy(c, row + j * m, multiplier, a, source, length)
                    }
                }
                inner = innerEnd
            }
            row = rowEnd
        }
        column = columnEnd
    }
}

/** Adds a symmetric rank-k product from an `n x depth` column-major operand to one triangle of C. */
@Suppress("LongParameterList")
internal fun blockedSyrkUpdate(
    kernels: F64Kernels,
    alpha: Double,
    a: DoubleArray,
    c: DoubleArray,
    n: Int,
    depth: Int,
    lower: Boolean,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + REFERENCE_NC, n)
        var row = if (lower) column else 0
        val rowLimit = if (lower) n else columnEnd
        while (row < rowLimit) {
            val rowEnd = min(row + REFERENCE_MC, n)
            var inner = 0
            while (inner < depth) {
                val innerEnd = min(inner + REFERENCE_KC, depth)
                for (p in inner until innerEnd) {
                    val sourceColumn = p * n
                    for (j in column until columnEnd) {
                        val from = maxOf(row, if (lower) j else 0)
                        val until = min(rowEnd, if (lower) n else j + 1)
                        if (from < until) {
                            val multiplier = alpha * a[j + sourceColumn]
                            if (multiplier != 0.0) {
                                kernels.axpy(c, from + j * n, multiplier, a, from + sourceColumn, until - from)
                            }
                        }
                    }
                }
                inner = innerEnd
            }
            row = rowEnd
        }
        column = columnEnd
    }
}

/** Adds a symmetric rank-2k product from two `n x depth` column-major operands to one triangle of C. */
@Suppress("LongParameterList")
internal fun blockedSyr2kUpdate(
    kernels: F64Kernels,
    alpha: Double,
    a: DoubleArray,
    b: DoubleArray,
    c: DoubleArray,
    n: Int,
    depth: Int,
    lower: Boolean,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + REFERENCE_NC, n)
        var row = if (lower) column else 0
        val rowLimit = if (lower) n else columnEnd
        while (row < rowLimit) {
            val rowEnd = min(row + REFERENCE_MC, n)
            var inner = 0
            while (inner < depth) {
                val innerEnd = min(inner + REFERENCE_KC, depth)
                for (p in inner until innerEnd) {
                    val sourceColumn = p * n
                    for (j in column until columnEnd) {
                        val from = maxOf(row, if (lower) j else 0)
                        val until = min(rowEnd, if (lower) n else j + 1)
                        if (from < until) {
                            val fromA = alpha * b[j + sourceColumn]
                            val fromB = alpha * a[j + sourceColumn]
                            if (fromA != 0.0) {
                                kernels.axpy(c, from + j * n, fromA, a, from + sourceColumn, until - from)
                            }
                            if (fromB != 0.0) {
                                kernels.axpy(c, from + j * n, fromB, b, from + sourceColumn, until - from)
                            }
                        }
                    }
                }
                inner = innerEnd
            }
            row = rowEnd
        }
        column = columnEnd
    }
}

/** Left-side SYMM, packing only the active symmetric panel before the common update loop. */
@Suppress("LongParameterList")
internal fun blockedSymmLeftUpdate(
    kernels: F64Kernels,
    alpha: Double,
    symmetric: DoubleArray,
    b: DoubleArray,
    c: DoubleArray,
    n: Int,
    columns: Int,
    lower: Boolean,
) {
    val panel = DoubleArray(REFERENCE_MC * REFERENCE_KC)
    var row = 0
    while (row < n) {
        val rowEnd = min(row + REFERENCE_MC, n)
        val rowCount = rowEnd - row
        var inner = 0
        while (inner < n) {
            val innerEnd = min(inner + REFERENCE_KC, n)
            val innerCount = innerEnd - inner
            for (p in 0 until innerCount) {
                val matrixColumn = inner + p
                for (i in 0 until rowCount) {
                    val matrixRow = row + i
                    val hi = maxOf(matrixRow, matrixColumn)
                    val lo = minOf(matrixRow, matrixColumn)
                    panel[i + p * rowCount] =
                        if (lower) symmetric[hi + lo * n] else symmetric[lo + hi * n]
                }
            }
            blockedUpdate(
                kernels,
                alpha,
                panel,
                0,
                rowCount,
                b,
                inner,
                n,
                c,
                row,
                n,
                rowCount,
                columns,
                innerCount,
            )
            inner = innerEnd
        }
        row = rowEnd
    }
}

/** Right-side SYMM; the ordinary left operand is contiguous and symmetric coefficients are scalar loads. */
@Suppress("LongParameterList")
internal fun blockedSymmRightUpdate(
    kernels: F64Kernels,
    alpha: Double,
    symmetric: DoubleArray,
    b: DoubleArray,
    c: DoubleArray,
    rows: Int,
    n: Int,
    lower: Boolean,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + REFERENCE_NC, n)
        var row = 0
        while (row < rows) {
            val rowEnd = min(row + REFERENCE_MC, rows)
            var inner = 0
            while (inner < n) {
                val innerEnd = min(inner + REFERENCE_KC, n)
                for (p in inner until innerEnd) {
                    for (j in column until columnEnd) {
                        val hi = maxOf(p, j)
                        val lo = minOf(p, j)
                        val av = if (lower) symmetric[hi + lo * n] else symmetric[lo + hi * n]
                        val multiplier = alpha * av
                        if (multiplier != 0.0) {
                            kernels.axpy(c, row + j * rows, multiplier, b, row + p * rows, rowEnd - row)
                        }
                    }
                }
                inner = innerEnd
            }
            row = rowEnd
        }
        column = columnEnd
    }
}

/** Packs `op(T)` while reading only T's selected triangle. */
internal fun packTriangularOp(
    source: DoubleArray,
    n: Int,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
): DoubleArray {
    val packed = DoubleArray(n * n)
    val effectiveLower = if (transpose) !lower else lower
    for (j in 0 until n) {
        val from = if (effectiveLower) j else 0
        val until = if (effectiveLower) n else j + 1
        for (i in from until until) {
            packed[i + j * n] = when {
                i == j && unitDiag -> 1.0
                transpose -> source[j + i * n]
                else -> source[i + j * n]
            }
        }
    }
    return packed
}
