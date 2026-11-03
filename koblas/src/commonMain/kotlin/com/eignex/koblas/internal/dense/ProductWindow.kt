@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.internal.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.portable.PortableEngine

/** Side of the square tile [PortableEngine] holds. Changing it means writing a different kernel. */
internal const val PORTABLE_PRODUCT_TILE: Int = 4

/**
 * `C = beta · C` over one product window, which is how a product block spends its destination multiplier.
 *
 * Spent once for the whole window before any tile accumulates into it, so a tile has one writeback and no
 * branch on the multiplier. A zero multiplier fills rather than multiplies: the destination's contract is
 * that its previous values take no part, not that they are multiplied by zero, and a NaN standing in the
 * output would survive the multiply.
 */
internal fun scaleProductWindow(beta: Double, c: DoubleArray, cOffset: Int, ldc: Int, rows: Int, columns: Int) {
    if (beta == 1.0) return
    for (column in 0 until columns) {
        val at = cOffset + column * ldc
        if (beta == 0.0) c.fill(0.0, at, at + rows) else for (i in 0 until rows) c[at + i] *= beta
    }
}

/**
 * Whether copying both operands into tiles pays for itself at these extents.
 *
 * The copy is `rows · depth + depth · columns` entries and the arithmetic `rows · columns · depth` products,
 * so what decides it is how much arithmetic each copied value takes part in. A product with an extent below
 * one tile cannot fill the tiles it would pack into, and a small product pays the copy against too little
 * arithmetic to hide it behind; both run as panel work over the operands where they are instead.
 *
 * The threshold reflects a measured crossover on one host. Sampled shapes favored packing above it,
 * while winners below it varied within the run-to-run band. Thin shapes can still lose above it;
 * no measured secondary condition separated those losses reliably.
 */
internal fun packsProductByWork(rows: Int, columns: Int, depth: Int, tileRows: Int, tileColumns: Int): Boolean =
    rows >= tileRows && columns >= tileColumns &&
        rows.toLong() * columns * depth >= PACKED_PRODUCT_MINIMUM_WORK

/** Products at or above this many multiply-adds are packed. */
internal const val PACKED_PRODUCT_MINIMUM_WORK: Long = 32L * 32L * 32L
