package com.eignex.koblas.internal.kernels

/**
 * Whether tiled operand copies amortize their cost. Sub-tile extents and small products use panels.
 * The threshold reflects one measured host: packing generally won above it, while winners below it
 * varied within run-to-run noise. Some thin shapes still lose above it; no reliable extra filter emerged.
 */
internal fun packsProductByWork(rows: Int, columns: Int, depth: Int, tileRows: Int, tileColumns: Int): Boolean =
    rows >= tileRows && columns >= tileColumns &&
        rows.toLong() * columns * depth >= PACKED_PRODUCT_MINIMUM_WORK

/** Products at or above this many multiply-adds are packed. */
internal const val PACKED_PRODUCT_MINIMUM_WORK: Long = 32L * 32L * 32L

/**
 * Applies `C = beta · C` once before tiles accumulate, keeping their writeback free of multiplier branches.
 * Zero beta overwrites without reading: multiplying would retain old NaNs and violate the no-read contract.
 */
internal fun scaleProductWindow(beta: Double, c: DoubleArray, cOffset: Int, ldc: Int, rows: Int, columns: Int) {
    if (beta == 1.0) return
    for (column in 0 until columns) {
        val at = cOffset + column * ldc
        if (beta == 0.0) c.fill(0.0, at, at + rows) else for (i in 0 until rows) c[at + i] *= beta
    }
}
