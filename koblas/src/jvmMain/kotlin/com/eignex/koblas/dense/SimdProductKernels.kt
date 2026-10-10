@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import jdk.incubator.vector.DoubleVector

/**
 * JVM Vector API product tile with portable fallbacks.
 *
 * Two lane blocks of rows and four columns reuse each broadcast coefficient twice and each left
 * vector four times. The shape depends on the register budget, independently of the lane count;
 * packers obtain it through [tileRows] and [tileColumns].
 *
 * Packed column groups have positive-zero padding, so partial columns use the full tile body with
 * fewer stores. Partial rows use the scalar edge because masked stores allocate on this
 * implementation. [implementationsFor] reports both bodies when needed.
 *
 * Species initialization requires the Vector API module. [com.eignex.koblas.BuiltinEngines] checks
 * availability before constructing an engine with this backend.
 */
internal object SimdProductKernels : DenseProductKernels {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = if (simdAvailable) SPECIES.length() else 0

    /** Two lane blocks where the module is present, and the portable tile's own rows where it is not. */
    private val ROWS = if (simdAvailable) ROW_BLOCKS * LANE else PORTABLE_PRODUCT_TILE

    /** Built once, for the reason [SimdPanelKernels] builds its own once. */
    private val NAME: String = "simd-tile(${ROWS}x$TILE_COLUMNS)"

    override val name: String get() = NAME

    override val tileRows: Int get() = ROWS

    override val tileColumns: Int get() = TILE_COLUMNS

    /**
     * The bodies a block of these extents reaches: the vector tile for whole lane-block rows, and the
     * scalar edge for the rows a last tile does not fill, in that order because that is the order the row
     * sweep below takes them in.
     */
    override fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> = when {
        !simdAvailable -> PortableProductKernels.implementationsFor(rows, columns, depth)
        rows <= 0 || columns <= 0 || depth <= 0 -> emptyList()
        rows < ROWS -> listOf(EDGE_NAME)
        rows % ROWS == 0 -> listOf(name)
        else -> listOf(name, EDGE_NAME)
    }

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
        if (!simdAvailable) {
            PortableProductKernels.productBlock(
                alpha, packedA, aOffset, aGroupStride, packedB, bOffset, bGroupStride,
                rows, columns, depth, beta, c, cOffset, ldc,
            )
            return
        }
        if (rows <= 0 || columns <= 0) return
        // Apply beta once: branched tile writeback prevents keeping accumulators in registers.
        scaleProductWindow(beta, c, cOffset, ldc, rows, columns)
        if (depth <= 0) return
        var column = 0
        while (column < columns) {
            val presentColumns = if (TILE_COLUMNS < columns - column) TILE_COLUMNS else columns - column
            val bPanel = bOffset + column / TILE_COLUMNS * bGroupStride
            var row = 0
            while (row < rows) {
                val presentRows = if (ROWS < rows - row) ROWS else rows - row
                val aPanel = aOffset + row / ROWS * aGroupStride
                val target = cOffset + row + column * ldc
                if (presentRows == ROWS) {
                    tile(depth, packedA, aPanel, packedB, bPanel, alpha, c, target, ldc, presentColumns)
                } else {
                    edgeTile(
                        depth, packedA, aPanel, packedB, bPanel, alpha, c, target, ldc,
                        presentRows, presentColumns,
                    )
                }
                row += ROWS
            }
            column += TILE_COLUMNS
        }
    }

    /**
     * Uses eight accumulators with hardware FMA, or two passes of four without it.
     *
     * The unfused eight-accumulator body allocates on the measured runtime; [columnPair] avoids that
     * allocation at the cost of reading the left panel twice. Both use the same packed geometry and
     * accumulate each entry in depth order. Fused arithmetic can round differently.
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
        presentColumns: Int,
    ) {
        if (!hardwareFusedMultiplyAdd) {
            columnPair(depth, a, aOffset, b, bOffset, 0, alpha, c, cOffset, ldc, presentColumns)
            if (presentColumns > COLUMN_PAIR) {
                columnPair(
                    depth, a, aOffset, b, bOffset, COLUMN_PAIR, alpha, c, cOffset + COLUMN_PAIR * ldc, ldc,
                    presentColumns - COLUMN_PAIR,
                )
            }
            return
        }
        fusedTile(depth, a, aOffset, b, bOffset, alpha, c, cOffset, ldc, presentColumns)
    }

    /**
     * Four accumulators avoid allocation in the unfused configuration. Each loaded left vector serves
     * both columns; partial columns accumulate padding without storing it.
     */
    private fun columnPair(
        depth: Int,
        a: DoubleArray,
        aOffset: Int,
        b: DoubleArray,
        bOffset: Int,
        firstColumn: Int,
        alpha: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
        presentColumns: Int,
    ) {
        var c00 = DoubleVector.zero(SPECIES)
        var c10 = DoubleVector.zero(SPECIES)
        var c01 = DoubleVector.zero(SPECIES)
        var c11 = DoubleVector.zero(SPECIES)
        var ap = aOffset
        var bp = bOffset + firstColumn
        var step = 0
        while (step < depth) {
            val a0 = DoubleVector.fromArray(SPECIES, a, ap)
            val a1 = DoubleVector.fromArray(SPECIES, a, ap + LANE)
            var coefficient = DoubleVector.broadcast(SPECIES, b[bp])
            c00 = multiplyAdd(a0, coefficient, c00)
            c10 = multiplyAdd(a1, coefficient, c10)
            coefficient = DoubleVector.broadcast(SPECIES, b[bp + 1])
            c01 = multiplyAdd(a0, coefficient, c01)
            c11 = multiplyAdd(a1, coefficient, c11)
            ap += ROWS
            bp += TILE_COLUMNS
            step++
        }
        storeColumn(c, cOffset, alpha, c00, c10)
        if (presentColumns > 1) storeColumn(c, cOffset + ldc, alpha, c01, c11)
    }

    private fun fusedTile(
        depth: Int,
        a: DoubleArray,
        aOffset: Int,
        b: DoubleArray,
        bOffset: Int,
        alpha: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
        presentColumns: Int,
    ) {
        var c00 = DoubleVector.zero(SPECIES)
        var c10 = DoubleVector.zero(SPECIES)
        var c01 = DoubleVector.zero(SPECIES)
        var c11 = DoubleVector.zero(SPECIES)
        var c02 = DoubleVector.zero(SPECIES)
        var c12 = DoubleVector.zero(SPECIES)
        var c03 = DoubleVector.zero(SPECIES)
        var c13 = DoubleVector.zero(SPECIES)
        var ap = aOffset
        var bp = bOffset
        var step = 0
        while (step < depth) {
            val a0 = DoubleVector.fromArray(SPECIES, a, ap)
            val a1 = DoubleVector.fromArray(SPECIES, a, ap + LANE)
            var coefficient = DoubleVector.broadcast(SPECIES, b[bp])
            c00 = multiplyAdd(a0, coefficient, c00)
            c10 = multiplyAdd(a1, coefficient, c10)
            coefficient = DoubleVector.broadcast(SPECIES, b[bp + 1])
            c01 = multiplyAdd(a0, coefficient, c01)
            c11 = multiplyAdd(a1, coefficient, c11)
            coefficient = DoubleVector.broadcast(SPECIES, b[bp + 2])
            c02 = multiplyAdd(a0, coefficient, c02)
            c12 = multiplyAdd(a1, coefficient, c12)
            coefficient = DoubleVector.broadcast(SPECIES, b[bp + 3])
            c03 = multiplyAdd(a0, coefficient, c03)
            c13 = multiplyAdd(a1, coefficient, c13)
            ap += ROWS
            bp += TILE_COLUMNS
            step++
        }
        storeColumn(c, cOffset, alpha, c00, c10)
        if (presentColumns > 1) storeColumn(c, cOffset + ldc, alpha, c01, c11)
        if (presentColumns > 2) storeColumn(c, cOffset + 2 * ldc, alpha, c02, c12)
        if (presentColumns > 3) storeColumn(c, cOffset + 3 * ldc, alpha, c03, c13)
    }

    /**
     * Scalar row edge, bounded by [tileRows] minus one rows.
     *
     * Packed inputs are padded, but destination columns end inside a lane block. Masked stores allocate
     * per vector on this implementation, so [implementationsFor] reports a scalar edge instead.
     */
    private fun edgeTile(
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
        for (column in 0 until presentColumns) {
            val target = cOffset + column * ldc
            for (row in 0 until presentRows) {
                var sum = 0.0
                var ap = aOffset + row
                var bp = bOffset + column
                var step = 0
                while (step < depth) {
                    sum += a[ap] * b[bp]
                    ap += ROWS
                    bp += TILE_COLUMNS
                    step++
                }
                c[target + row] += alpha * sum
            }
        }
    }

    /** One whole destination column of a tile, with every lane of both blocks inside the window. */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun storeColumn(c: DoubleArray, at: Int, alpha: Double, low: DoubleVector, high: DoubleVector) {
        store(c, at, alpha, low)
        store(c, at + LANE, alpha, high)
    }

    /**
     * `c += alpha · accumulated` over a whole lane block.
     *
     * Inline to keep [DoubleVector] arguments in registers rather than materializing heap objects.
     * A branch at writeback also prevents allocation elimination, so beta is applied before the tiles.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun store(c: DoubleArray, at: Int, alpha: Double, accumulated: DoubleVector) {
        DoubleVector.fromArray(SPECIES, c, at).add(accumulated.mul(alpha)).intoArray(c, at)
    }

    /**
     * One multiply-add, fused where [hardwareFusedMultiplyAdd] found the instruction and two operations
     * where it did not.
     *
     * Inline because a helper returning a [DoubleVector] that the JIT declines to inline makes the vector a
     * heap object, which turns a register-resident tile into an allocation per depth step.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun multiplyAdd(x: DoubleVector, y: DoubleVector, accumulator: DoubleVector): DoubleVector =
        if (hardwareFusedMultiplyAdd) x.fma(y, accumulator) else x.mul(y).add(accumulator)

    /** Lane blocks of destination rows one tile holds. */
    private const val ROW_BLOCKS = 2

    /** Destination columns one tile holds, which is a register budget and not a lane count. */
    private const val TILE_COLUMNS = 4

    /** Destination columns one pass covers where a step takes two instructions instead of one. */
    private const val COLUMN_PAIR = 2

    /** What the scalar last row block reports, so a vector body's name never stands for it. */
    private const val EDGE_NAME = "scalar-tile-edge"
}
