package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import kotlin.math.min

/*
 * Cache tiles for the portable level-3 routines: a block of the product covers a fixed number of rows and
 * columns and is accumulated over a fixed depth of the shared dimension at a time. Why each has the value
 * it has, and how to change one without rebuilding, is on the entry in [DenseTuning]. They are bound to
 * names here because that is what the loops below read, and binding them once keeps the read off the block
 * loops.
 */
internal val LEVEL3_BLOCK_ROWS: Int = DenseTuning.level3BlockRows
internal val LEVEL3_BLOCK_COLUMNS: Int = DenseTuning.level3BlockColumns
internal val LEVEL3_BLOCK_DEPTH: Int = DenseTuning.level3BlockDepth

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
    kernels: DensePanelKernels,
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
    kernels: DensePanelKernels,
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
        val columnEnd = min(column + LEVEL3_BLOCK_COLUMNS, n)
        var inner = 0
        while (inner < depth) {
            val innerEnd = min(inner + LEVEL3_BLOCK_DEPTH, depth)
            // The row block is outside the two operand loops, which is what makes it a block: everything
            // below it touches A's rows `row until row + length` and C's, and nothing else. With the row
            // loop innermost the live piece of C was the whole of `m x NC` however small the tile was, so
            // the tile bounded no working set and only chopped one axpy into several.
            var row = 0
            while (row < m) {
                val length = min(row + LEVEL3_BLOCK_ROWS, m) - row
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

/**
 * Adds `alpha * A transpose * B` to C without packing A. Columns of A and B are contiguous dot operands;
 * four output rows share each B column through [DensePanelKernels.dot4].
 */
@Suppress("LongParameterList")
internal fun blockedTransposedLeftUpdate(
    panelKernels: DensePanelKernels,
    vectorKernels: DenseVectorKernels,
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
        val columnEnd = min(column + LEVEL3_BLOCK_COLUMNS, n)
        var row = 0
        while (row < m) {
            val rowEnd = min(row + LEVEL3_BLOCK_ROWS, m)
            // The row quad is the outer of the two, so the four A columns it reads stay in L1 while the
            // column block's B streams past them. With the columns outside instead, each of them re-read
            // the whole row panel, which is LEVEL3_BLOCK_COLUMNS passes over it rather than one.
            var i = row
            while (i + 4 <= rowEnd) {
                for (j in column until columnEnd) {
                    var inner = 0
                    while (inner < depth) {
                        val length = min(inner + LEVEL3_BLOCK_DEPTH, depth) - inner
                        dot4Writeback(
                            panelKernels, alpha, a, aOff + inner + i * lda, lda,
                            b, bOff + inner + j * ldb, length, c, cOff + i + j * ldc, sums,
                        )
                        inner += length
                    }
                }
                i += 4
            }
            while (i < rowEnd) {
                for (j in column until columnEnd) {
                    var inner = 0
                    while (inner < depth) {
                        val length = min(inner + LEVEL3_BLOCK_DEPTH, depth) - inner
                        dotWriteback(
                            vectorKernels, alpha, a, aOff + inner + i * lda,
                            b, bOff + inner + j * ldb, length, c, cOff + i + j * ldc,
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
    kernels: DensePanelKernels,
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
    kernels: DensePanelKernels,
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
    kernels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    b: DoubleArray,
    c: DoubleArray,
    n: Int,
    depth: Int,
    lower: Boolean,
    guardZeroColumns: Boolean = true,
): Unit = blockedSymmetricRankUpdate(kernels, alpha, a, b, c, n, depth, lower, guardZeroColumns)
