@file:Suppress("LongParameterList") // Independent windows and extents.

package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.borrowOptional

/*
 * Symmetric products use shared product scheduling over diagonal blocks and stored off-diagonal strips.
 * Each strip serves both halves, once directly and once transposed; only diagonal blocks expand into scratch.
 * Nothing outside the stored triangle is read. Beta applies once to the full destination before accumulation.
 */

/**
 * Diagonal block order, independent of register tiles, panel groups and cache blocks.
 * It bounds the expanded scratch square and trades copy size against the number of strip products.
 */
internal const val SYMMETRIC_BLOCK: Int = 64

/**
 * The diagonal blocks of a symmetric operand and the stored strip beside each, in traversal order.
 *
 * The strip is the part of the stored triangle that the block's columns reach: the rows below it where the
 * lower triangle is stored, and the rows above it where the upper one is.
 */
internal inline fun forEachSymmetricBlock(
    order: Int,
    block: Int,
    lower: Boolean,
    action: (start: Int, size: Int, stripStart: Int, stripCount: Int) -> Unit,
) {
    var start = 0
    while (start < order) {
        val size = if (block < order - start) block else order - start
        val end = start + size
        val stripStart = if (lower) end else 0
        val stripCount = if (lower) order - end else start
        action(start, size, stripStart, stripCount)
        start = end
    }
}

/**
 * `C = alpha · A · B + beta · C`, or the same with the operands the other way round when [right].
 *
 * [a] is the symmetric operand of order [order] and [b] the dense one, both where the caller put them and
 * both already staged against an overlap with [c].
 */
internal fun symmetricProduct(
    engine: KoblasEngine,
    alpha: Double,
    a: DoubleArray,
    order: Int,
    lower: Boolean,
    b: DoubleArray,
    ldb: Int,
    c: DoubleArray,
    ldc: Int,
    rows: Int,
    columns: Int,
    right: Boolean,
    arrays: Scratch?,
    aOffset: Int = 0,
    lda: Int = order,
    bOffset: Int = 0,
    cOffset: Int = 0,
) {
    // One window with op(S), a piece of the symmetric operand spanning these steps of the order down and across.
    // Its rows are the shared dimension when the dense operand is on the left, and the destination's otherwise.
    fun window(
        s: DoubleArray,
        at: Int,
        lds: Int,
        transposeS: Boolean,
        rowStart: Int,
        opRows: Int,
        columnStart: Int,
        opColumns: Int,
    ) {
        val storedRows = if (transposeS) opColumns else opRows
        val storedColumns = if (transposeS) opRows else opColumns
        val symmetric = DenseMatrix.wrap(storedRows, storedColumns, s, at, lds)
        if (right) {
            productWindow(
                engine, alpha, DenseMatrix.wrap(rows, opRows, b, bOffset + rowStart * ldb, ldb), false,
                symmetric, transposeS, 1.0,
                DenseMatrix.wrap(rows, opColumns, c, cOffset + columnStart * ldc, ldc),
                OutputTriangle.Full, arrays,
            )
        } else {
            productWindow(
                engine, alpha, symmetric, transposeS,
                DenseMatrix.wrap(opColumns, columns, b, bOffset + columnStart, ldb), false, 1.0,
                DenseMatrix.wrap(opRows, columns, c, cOffset + rowStart, ldc),
                OutputTriangle.Full, arrays,
            )
        }
    }

    val block = if (SYMMETRIC_BLOCK < order) SYMMETRIC_BLOCK else order
    arrays.borrowOptional(block * block) { square ->
        forEachSymmetricBlock(order, block, lower) { start, size, stripStart, stripCount ->
            expandSymmetric(a, lda, lower, start, size, square, aOffset)
            window(square, 0, size, false, start, size, start, size)
            if (stripCount > 0) {
                val strip = aOffset + stripStart + start * lda
                window(a, strip, lda, false, stripStart, stripCount, start, size)
                window(a, strip, lda, true, start, size, stripStart, stripCount)
            }
        }
    }
}

/**
 * Expand one diagonal block for ordinary dense products, mirroring only the stored triangle.
 * Scratch is bounded by the block size.
 */
private fun expandSymmetric(
    a: DoubleArray,
    order: Int,
    lower: Boolean,
    start: Int,
    size: Int,
    square: DoubleArray,
    offset: Int,
) {
    for (column in 0 until size) {
        for (row in 0 until size) {
            val stored = if (lower == (row >= column)) {
                a[offset + start + row + (start + column) * order]
            } else {
                a[offset + start + column + (start + row) * order]
            }
            square[row + column * size] = stored
        }
    }
}
