package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import kotlin.math.min

/** Cache tiles for the portable level-3 routines. They are deliberately target-neutral starting values. */
internal const val REFERENCE_MC: Int = 256
internal const val REFERENCE_NC: Int = 8
internal const val REFERENCE_KC: Int = 128
/**
 * Diagonal block width for the blocked triangular routines.
 *
 * Not free to retune: the trsm zero-pivot guard carries one bit per row of a block in a `Long` mask, and
 * Kotlin's `shl` reads only the low six bits of its operand, so a width above [Long.SIZE_BITS] would wrap
 * row `j` onto bit `j % 64` and mask the wrong rows. [requireTriangularBlockFitsMask] holds that link.
 */
internal const val REFERENCE_TRIANGULAR_BLOCK: Int = 64
private const val REFERENCE_TRANSPOSE_BLOCK: Int = 32

/**
 * Fails when [REFERENCE_TRIANGULAR_BLOCK] outgrows the zero-pivot mask that indexes it.
 *
 * A wrapped shift would not throw or read out of bounds. It would quietly retain the products of one row
 * against the mask bit of another, which only shows up as a wrong answer on a matrix whose quotient
 * underflows, so the check has to be explicit rather than left to a test to notice.
 */
internal fun requireTriangularBlockFitsMask() {
    require(REFERENCE_TRIANGULAR_BLOCK <= Long.SIZE_BITS) {
        "REFERENCE_TRIANGULAR_BLOCK is $REFERENCE_TRIANGULAR_BLOCK, above the ${Long.SIZE_BITS} rows a " +
            "Long zero-pivot mask can index"
    }
}

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

/** Borrows a `[rows]x[cols]` buffer and packs it with the transpose of [src], for a caller that only ever
 *  wants the transposed layout on scratch. */
internal inline fun <T> Workspace?.borrowTransposed(
    src: DoubleArray,
    rows: Int,
    cols: Int,
    block: (DoubleArray) -> T,
): T = borrow(rows * cols) { packed ->
    transposeBlocked(src, rows, cols, packed)
    block(packed)
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
    skipZeroCoefficient: Boolean = true,
    zeroCoefficientMasks: LongArray? = null,
): Unit = blockedAxpyUpdate(
    kernels, alpha, a, aOff, lda, c, cOff, ldc, m, n, depth, skipZeroCoefficient, zeroCoefficientMasks,
) { p, j ->
    b[bOff + p + j * ldb]
}

/** Shared cache traversal for products whose left operand has contiguous columns. */
@Suppress("LongParameterList")
private inline fun blockedAxpyUpdate(
    kernels: F64Kernels,
    alpha: Double,
    a: DoubleArray,
    aOff: Int,
    lda: Int,
    c: DoubleArray,
    cOff: Int,
    ldc: Int,
    m: Int,
    n: Int,
    depth: Int,
    skipZeroCoefficient: Boolean = true,
    zeroCoefficientMasks: LongArray? = null,
    coefficient: (p: Int, j: Int) -> Double,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + REFERENCE_NC, n)
        var inner = 0
        while (inner < depth) {
            val innerEnd = min(inner + REFERENCE_KC, depth)
            for (p in inner until innerEnd) {
                for (j in column until columnEnd) {
                    val value = coefficient(p, j)
                    val forcedZero = zeroCoefficientMasks != null && zeroCoefficientMasks[j] and (1L shl p) != 0L
                    if (!skipZeroCoefficient || value != 0.0 || forcedZero) {
                        val multiplier = alpha * value
                        var row = 0
                        while (row < m) {
                            val length = min(row + REFERENCE_MC, m) - row
                            val source = aOff + row + p * lda
                            axpyArithmetic(kernels, c, cOff + row + j * ldc, multiplier, a, source, length)
                            row += length
                        }
                    }
                }
            }
            inner = innerEnd
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
    skipZeroCoefficient: Boolean,
) {
    if (!transposeB) {
        blockedAxpyUpdate(
            kernels, alpha, a, 0, m, c, 0, m, m, n, depth, skipZeroCoefficient,
        ) { p, j ->
            b[p + j * bRows]
        }
        return
    }
    blockedAxpyUpdate(
        kernels, alpha, a, 0, m, c, 0, m, m, n, depth, skipZeroCoefficient,
    ) { p, j ->
        b[j + p * bRows]
    }
}

/**
 * Adds `alpha * A transpose * B` to C without packing A. Columns of A and B are contiguous dot operands;
 * four output rows share each B column through [F64Kernels.dot4].
 */
@Suppress("LongParameterList")
internal fun blockedTransposedLeftUpdate(
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
    sums: DoubleArray,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + REFERENCE_NC, n)
        var row = 0
        while (row < m) {
            val rowEnd = min(row + REFERENCE_MC, m)
            for (j in column until columnEnd) {
                var i = row
                while (i + 4 <= rowEnd) {
                    var inner = 0
                    while (inner < depth) {
                        val length = min(inner + REFERENCE_KC, depth) - inner
                        kernels.dot4(
                            a,
                            aOff + inner + i * lda,
                            lda,
                            b,
                            bOff + inner + j * ldb,
                            length,
                            sums,
                            0,
                        )
                        for (r in 0 until 4) c[cOff + i + r + j * ldc] += alpha * sums[r]
                        inner += length
                    }
                    i += 4
                }
                while (i < rowEnd) {
                    var inner = 0
                    while (inner < depth) {
                        val length = min(inner + REFERENCE_KC, depth) - inner
                        c[cOff + i + j * ldc] += alpha * kernels.dot(
                            a,
                            aOff + inner + i * lda,
                            b,
                            bOff + inner + j * ldb,
                            length,
                        )
                        inner += length
                    }
                    i++
                }
            }
            row = rowEnd
        }
        column = columnEnd
    }
}

/** Adds a right-side triangular panel product, reading `op(T)` without packing a transposed triangle. */
@Suppress("LongParameterList")
internal fun blockedRightTriangularUpdate(
    kernels: F64Kernels,
    alpha: Double,
    b: DoubleArray,
    rows: Int,
    triangle: DoubleArray,
    triangleOrder: Int,
    transpose: Boolean,
    innerStart: Int,
    depth: Int,
    columnStart: Int,
    columns: Int,
) = blockedAxpyUpdate(
    kernels,
    alpha,
    b,
    innerStart * rows,
    rows,
    b,
    columnStart * rows,
    rows,
    rows,
    columns,
    depth,
    zeroCoefficientMasks = null,
) { p, j ->
    val factorRow = innerStart + p
    val factorColumn = columnStart + j
    if (transpose) {
        triangle[factorColumn + factorRow * triangleOrder]
    } else {
        triangle[factorRow + factorColumn * triangleOrder]
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
    guardZeroColumns: Boolean = true,
): Unit = blockedSymmetricRankUpdate(kernels, alpha, a, null, c, n, depth, lower, guardZeroColumns)

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
    guardZeroColumns: Boolean = true,
): Unit = blockedSymmetricRankUpdate(kernels, alpha, a, b, c, n, depth, lower, guardZeroColumns)

/** Shared cache traversal for rank-k and rank-2k updates. */
@Suppress("LongParameterList")
private fun blockedSymmetricRankUpdate(
    kernels: F64Kernels,
    alpha: Double,
    a: DoubleArray,
    b: DoubleArray?,
    c: DoubleArray,
    n: Int,
    depth: Int,
    lower: Boolean,
    guardZeroColumns: Boolean,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + REFERENCE_NC, n)
        var inner = 0
        while (inner < depth) {
            val innerEnd = min(inner + REFERENCE_KC, depth)
            for (p in inner until innerEnd) {
                val sourceColumn = p * n
                for (j in column until columnEnd) {
                    val firstValue = b?.get(j + sourceColumn) ?: a[j + sourceColumn]
                    val secondValue = if (b == null) 0.0 else a[j + sourceColumn]
                    val skip = guardZeroColumns && if (b == null) {
                        firstValue == 0.0
                    } else {
                        firstValue == 0.0 && secondValue == 0.0
                    }
                    if (skip) continue
                    val firstMultiplier = alpha * firstValue
                    val secondMultiplier = alpha * secondValue
                    val triangleFrom = if (lower) j else 0
                    val triangleUntil = if (lower) n else j + 1
                    var row = triangleFrom
                    while (row < triangleUntil) {
                        val length = min(row + REFERENCE_MC, triangleUntil) - row
                        axpyArithmetic(kernels, c, row + j * n, firstMultiplier, a, row + sourceColumn, length)
                        if (b != null) {
                            axpyArithmetic(kernels, c, row + j * n, secondMultiplier, b, row + sourceColumn, length)
                        }
                        row += length
                    }
                }
            }
            inner = innerEnd
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
    panel: DoubleArray,
) {
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
                skipZeroCoefficient = false,
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
): Unit = blockedAxpyUpdate(
    kernels, alpha, b, 0, rows, c, 0, rows, rows, n, n, false,
) { p, j ->
    val hi = maxOf(p, j)
    val lo = minOf(p, j)
    if (lower) symmetric[hi + lo * n] else symmetric[lo + hi * n]
}
