@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.dense

/**
 * The portable product tile every platform has, and the floor a vector backend falls back to.
 *
 * Four destination rows by four columns in sixteen scalar accumulators. Separate locals are the point: an
 * accumulator array would put the tile back in memory, and the whole reason to hold a rectangle of the
 * destination rather than one entry is that the shared dimension is then walked once for sixteen products
 * instead of once for each.
 *
 * Four by four is this file's own geometry and appears in no algorithm above it. It is not the panel
 * execution group, which groups columns of a Level 2 window, and it is not a cache block, which is the
 * shared scheduling's choice. A caller learns it from [tileRows] and [tileColumns] when it packs, which is
 * the only place the number is needed.
 */
internal object PortableProductKernels : DenseProductKernels {
    private val NAME: String = "scalar-tile(${PORTABLE_PRODUCT_TILE}x$PORTABLE_PRODUCT_TILE)"

    override val name: String get() = NAME

    override val tileRows: Int get() = PORTABLE_PRODUCT_TILE

    override val tileColumns: Int get() = PORTABLE_PRODUCT_TILE

    override fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> =
        if (rows <= 0 || columns <= 0 || depth <= 0) emptyList() else listOf(name)

    override fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean =
        packsProductByWork(rows, columns, depth, tileRows, tileColumns)

    override fun productBlock(
        alpha: Double,
        packedA: DoubleArray,
        aOffset: Int,
        aGroupStride: Int,
        packedB: DoubleArray,
        bOffset: Int,
        bGroupStride: Int,
        rows: Int,
        columns: Int,
        depth: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        scaleProductWindow(beta, c, cOffset, ldc, rows, columns)
        if (depth <= 0) return
        var column = 0
        while (column < columns) {
            val presentColumns = if (PORTABLE_PRODUCT_TILE <
                columns - column
            ) {
                PORTABLE_PRODUCT_TILE
            } else {
                columns - column
            }
            val bPanel = bOffset + column / PORTABLE_PRODUCT_TILE * bGroupStride
            var row = 0
            while (row < rows) {
                val presentRows = if (PORTABLE_PRODUCT_TILE < rows - row) PORTABLE_PRODUCT_TILE else rows - row
                val aPanel = aOffset + row / PORTABLE_PRODUCT_TILE * aGroupStride
                tile(
                    depth, packedA, aPanel, packedB, bPanel, alpha,
                    c, cOffset + row + column * ldc, ldc, presentRows, presentColumns,
                )
                row += PORTABLE_PRODUCT_TILE
            }
            column += PORTABLE_PRODUCT_TILE
        }
    }

    /**
     * One four by four tile, accumulated over the whole depth and written back once.
     *
     * Padding lanes of a short edge accumulate zeros, because a packed group pads with positive zero, and
     * are then not stored: [presentRows] and [presentColumns] are what reaches the destination.
     */
    private fun tile(
        depth: Int,
        a: DoubleArray,
        aOffset: Int,
        b: DoubleArray,
        bOffset: Int,
        alpha: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
        presentRows: Int,
        presentColumns: Int,
    ) {
        var c00 = 0.0
        var c10 = 0.0
        var c20 = 0.0
        var c30 = 0.0
        var c01 = 0.0
        var c11 = 0.0
        var c21 = 0.0
        var c31 = 0.0
        var c02 = 0.0
        var c12 = 0.0
        var c22 = 0.0
        var c32 = 0.0
        var c03 = 0.0
        var c13 = 0.0
        var c23 = 0.0
        var c33 = 0.0
        var ap = aOffset
        var bp = bOffset
        var step = 0
        while (step < depth) {
            val a0 = a[ap]
            val a1 = a[ap + 1]
            val a2 = a[ap + 2]
            val a3 = a[ap + 3]
            var coefficient = b[bp]
            c00 += a0 * coefficient
            c10 += a1 * coefficient
            c20 += a2 * coefficient
            c30 += a3 * coefficient
            coefficient = b[bp + 1]
            c01 += a0 * coefficient
            c11 += a1 * coefficient
            c21 += a2 * coefficient
            c31 += a3 * coefficient
            coefficient = b[bp + 2]
            c02 += a0 * coefficient
            c12 += a1 * coefficient
            c22 += a2 * coefficient
            c32 += a3 * coefficient
            coefficient = b[bp + 3]
            c03 += a0 * coefficient
            c13 += a1 * coefficient
            c23 += a2 * coefficient
            c33 += a3 * coefficient
            ap += PORTABLE_PRODUCT_TILE
            bp += PORTABLE_PRODUCT_TILE
            step++
        }
        storeColumn(c, cOffset, presentRows, alpha, c00, c10, c20, c30)
        if (presentColumns > 1) storeColumn(c, cOffset + ldc, presentRows, alpha, c01, c11, c21, c31)
        if (presentColumns > 2) storeColumn(c, cOffset + 2 * ldc, presentRows, alpha, c02, c12, c22, c32)
        if (presentColumns > 3) storeColumn(c, cOffset + 3 * ldc, presentRows, alpha, c03, c13, c23, c33)
    }

    private fun storeColumn(
        c: DoubleArray,
        at: Int,
        presentRows: Int,
        alpha: Double,
        v0: Double,
        v1: Double,
        v2: Double,
        v3: Double,
    ) {
        c[at] += alpha * v0
        if (presentRows > 1) c[at + 1] += alpha * v1
        if (presentRows > 2) c[at + 2] += alpha * v2
        if (presentRows > 3) c[at + 3] += alpha * v3
    }
}

/** Side of the square tile [PortableProductKernels] holds. Changing it means writing a different kernel. */
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
 * The limit is where the copy starts winning consistently rather than where it first wins, and it remains
 * one machine's crossover. The local evidence compares the two schedules over the same operands: the
 * sampled shapes favored packing above this threshold, and below it the winner varied within the
 * run-to-run band. The calibration also looked for a second condition to put beside this one and found
 * none defensible, since no quantity it measured separated the shapes packing loses on, thin ones above
 * all, from the ones it wins. The threshold is therefore unchanged, and the shapes where it costs
 * something on that host are recorded with the local evidence rather than fitted to.
 */
internal fun packsProductByWork(rows: Int, columns: Int, depth: Int, tileRows: Int, tileColumns: Int): Boolean =
    rows >= tileRows && columns >= tileColumns &&
        rows.toLong() * columns * depth >= PACKED_PRODUCT_MINIMUM_WORK

/** Products at or above this many multiply-adds are packed. */
internal const val PACKED_PRODUCT_MINIMUM_WORK: Long = 32L * 32L * 32L
