@file:Suppress("LongParameterList") // a symmetric product carries both operands, the side and the triangle

package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*

/*
 * `C = alpha · A · B + beta · C` and `C = alpha · B · A + beta · C` for a symmetric `A`, as ordinary windows
 * of shared product scheduling.
 *
 * A symmetric operand is a square whose stored triangle stands for both halves, which is the one thing a
 * product block cannot be told. So the square is cut instead: the blocks along its diagonal, and the strips
 * of the stored triangle beside them. A strip is an ordinary rectangle of stored entries and is used twice,
 * once as itself and once transposed, which is what the mirrored half of the matrix is. Nothing outside the
 * stored triangle is read, and nothing is expanded except one diagonal block at a time.
 *
 * A diagonal block is the part that has no rectangular form: it holds the diagonal, where the two halves
 * meet. It is written out into a square of scratch bounded by the block size, and multiplied as an ordinary
 * dense window. The copy is one block of the order per block of the order, which against a product over
 * every right-hand side is a pass over the operand rather than a pass over the work.
 *
 * `beta` is spent once, on the whole destination, before any of this: the destination of a symmetric product
 * is written in full, and each of its entries is accumulated into by several of the windows below.
 */

/**
 * Steps of the order one diagonal block of a symmetric operand covers.
 *
 * This file's own number, independent of the register tile, the panel grouping and the cache block. It
 * bounds the scratch a symmetric product expands into, which is one square of it, and it sets how often the
 * strips beside the diagonal are cut: a smaller block copies less and schedules more products, a larger one
 * the reverse.
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
        if (right) {
            productWindow(
                engine, alpha, b, bOffset + rowStart * ldb, ldb, false, s, at, lds, transposeS, 1.0, c,
                cOffset + columnStart * ldc, ldc, rows, opColumns, opRows, OutputTriangle.Full, arrays,
            )
        } else {
            productWindow(
                engine, alpha, s, at, lds, transposeS, b, bOffset + columnStart, ldb, false, 1.0, c,
                cOffset + rowStart, ldc, opRows, columns, opColumns, OutputTriangle.Full, arrays,
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
 * One diagonal block of the symmetric operand written out as a full square.
 *
 * Only the stored triangle is read. The entry above a lower triangle's diagonal is its mirror below, which
 * is the whole of what a symmetric operand means, and the copy is what lets an ordinary product block read
 * the block without being told about the structure.
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
