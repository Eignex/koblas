package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import kotlin.math.min

/*
 * Cache tiles for the portable level-3 routines: a block of the product is `MC` rows by `NC` columns and is
 * accumulated over `KC` of the shared dimension at a time. Why each has the value it has, and how to change
 * one without rebuilding, is on the entry in [DenseTuning]. They are bound to names here because that is
 * what the loops below read, and binding them once keeps the read off the block loops.
 */
internal val REFERENCE_MC: Int = DenseTuning.level3BlockRows
internal val REFERENCE_NC: Int = DenseTuning.level3BlockColumns
internal val REFERENCE_KC: Int = DenseTuning.level3BlockDepth

/** Diagonal block width for the blocked triangular routines, bounded by the mask that indexes it. */
internal val REFERENCE_TRIANGULAR_BLOCK: Int = DenseTuning.triangularBlock
private val REFERENCE_TRANSPOSE_BLOCK: Int = DenseTuning.transposeBlock

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
            // The row block is outside the two operand loops, which is what makes it a block: everything
            // below it touches A's rows `row until row + length` and C's, and nothing else. With the row
            // loop innermost the live piece of C was the whole of `m x NC` however small the tile was, so
            // the tile bounded no working set and only chopped one axpy into several.
            var row = 0
            while (row < m) {
                val length = min(row + REFERENCE_MC, m) - row
                for (p in inner until innerEnd) {
                    val source = aOff + row + p * lda
                    for (j in column until columnEnd) {
                        val value = coefficient(p, j)
                        val forcedZero = zeroCoefficientMasks != null && zeroCoefficientMasks[j] and (1L shl p) != 0L
                        if (!skipZeroCoefficient || value != 0.0 || forcedZero) {
                            axpyArithmetic(kernels, c, cOff + row + j * ldc, alpha * value, a, source, length)
                        }
                    }
                }
                row += length
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
            // The row quad is the outer of the two, so the four A columns it reads stay in L1 while the
            // column block's B streams past them. With the columns outside instead, each of them re-read
            // the whole row panel, which is REFERENCE_NC passes over it rather than one.
            var i = row
            while (i + 4 <= rowEnd) {
                for (j in column until columnEnd) {
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
                }
                i += 4
            }
            while (i < rowEnd) {
                for (j in column until columnEnd) {
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
                }
                i++
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
