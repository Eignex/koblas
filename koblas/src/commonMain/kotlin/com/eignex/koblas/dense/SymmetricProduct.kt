@file:Suppress("LongParameterList") // a symmetric product carries both operands, the side and the triangle

package com.eignex.koblas.dense

import com.eignex.koblas.Workspace

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
 * lower triangle is stored, and the rows above it where the upper one is. One traversal, walked by the
 * execution and by the route that describes it.
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
    products: DenseProductKernels,
    panels: DensePanelKernels,
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
    workspace: Workspace?,
) {
    if (order == 0 || rows == 0 || columns == 0) return
    val block = if (SYMMETRIC_BLOCK < order) SYMMETRIC_BLOCK else order
    workspace.borrowOptional(block * block) { square ->
        forEachSymmetricBlock(order, block, lower) { start, size, stripStart, stripCount ->
            expandSymmetric(a, order, lower, start, size, square)
            if (right) {
                rightSymmetricBlock(
                    products, panels, alpha, a, order, b, ldb, c, ldc, rows,
                    start, size, stripStart, stripCount, square, workspace,
                )
            } else {
                leftSymmetricBlock(
                    products, panels, alpha, a, order, b, ldb, c, ldc, columns,
                    start, size, stripStart, stripCount, square, workspace,
                )
            }
        }
    }
}

/** One diagonal block and its strip, with the symmetric operand on the left of the product. */
private fun leftSymmetricBlock(
    products: DenseProductKernels,
    panels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    order: Int,
    b: DoubleArray,
    ldb: Int,
    c: DoubleArray,
    ldc: Int,
    columns: Int,
    start: Int,
    size: Int,
    stripStart: Int,
    stripCount: Int,
    square: DoubleArray,
    workspace: Workspace?,
) {
    productWindow(
        products, panels, alpha, square, 0, size, false, b, start, ldb, false,
        1.0, c, start, ldc, size, columns, size, OutputTriangle.Full, workspace,
    )
    if (stripCount == 0) return
    val strip = stripStart + start * order
    // The strip as it is stored: the rows beside this block accumulate the block's right-hand sides.
    productWindow(
        products, panels, alpha, a, strip, order, false, b, start, ldb, false,
        1.0, c, stripStart, ldc, stripCount, columns, size, OutputTriangle.Full, workspace,
    )
    // The same strip transposed, which is the mirrored half: the block's rows accumulate the strip's.
    productWindow(
        products, panels, alpha, a, strip, order, true, b, stripStart, ldb, false,
        1.0, c, start, ldc, size, columns, stripCount, OutputTriangle.Full, workspace,
    )
}

/** One diagonal block and its strip, with the symmetric operand on the right of the product. */
private fun rightSymmetricBlock(
    products: DenseProductKernels,
    panels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    order: Int,
    b: DoubleArray,
    ldb: Int,
    c: DoubleArray,
    ldc: Int,
    rows: Int,
    start: Int,
    size: Int,
    stripStart: Int,
    stripCount: Int,
    square: DoubleArray,
    workspace: Workspace?,
) {
    productWindow(
        products, panels, alpha, b, start * ldb, ldb, false, square, 0, size, false,
        1.0, c, start * ldc, ldc, rows, size, size, OutputTriangle.Full, workspace,
    )
    if (stripCount == 0) return
    val strip = stripStart + start * order
    // The mirrored half first: this block's columns of the destination take the strip's columns of B.
    productWindow(
        products, panels, alpha, b, stripStart * ldb, ldb, false, a, strip, order, false,
        1.0, c, start * ldc, ldc, rows, size, stripCount, OutputTriangle.Full, workspace,
    )
    // Then the strip as it is stored, whose rows are the destination columns beside this block.
    productWindow(
        products, panels, alpha, b, start * ldb, ldb, false, a, strip, order, true,
        1.0, c, stripStart * ldc, ldc, rows, stripCount, size, OutputTriangle.Full, workspace,
    )
}

/**
 * One diagonal block of the symmetric operand written out as a full square.
 *
 * Only the stored triangle is read. The entry above a lower triangle's diagonal is its mirror below, which
 * is the whole of what a symmetric operand means, and the copy is what lets an ordinary product block read
 * the block without being told about the structure.
 */
private fun expandSymmetric(a: DoubleArray, order: Int, lower: Boolean, start: Int, size: Int, square: DoubleArray) {
    for (column in 0 until size) {
        for (row in 0 until size) {
            val stored = if (lower == (row >= column)) {
                a[start + row + (start + column) * order]
            } else {
                a[start + column + (start + row) * order]
            }
            square[row + column * size] = stored
        }
    }
}
