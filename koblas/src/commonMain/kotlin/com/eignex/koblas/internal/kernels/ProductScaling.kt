package com.eignex.koblas.internal.kernels

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
