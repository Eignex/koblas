@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import jdk.incubator.vector.DoubleVector

/**
 * The JVM Vector API product tile, with the portable tile underneath it.
 *
 * Two lane blocks of destination rows by four destination columns, held in eight vector accumulators for the
 * whole depth of a block. Two lane blocks of rows is what makes each broadcast coefficient serve two vector
 * multiply-adds, and four columns is what makes each loaded left vector serve four. How many live vectors a
 * machine can hold before it starts spilling is its own architecture's answer and not something a lane count
 * settles, which is why the shape is a measurement rather than a calculation.
 *
 * The stage evidence holds that measurement: six rectangles, each written out with its own named
 * accumulators and each allocation-free, over the same logical product at several shapes on one machine.
 * This one led at every one of them. That is one machine's answer, and a machine with a different register
 * file or cache may want another, which is why the geometry is read from [tileRows] and [tileColumns] by
 * everything that packs for it rather than written into a caller.
 *
 * A destination whose columns run out part way through a tile is the same body with fewer columns stored,
 * since the packed groups are full and their padding is positive zero. A destination whose *rows* run out
 * is not: that would need a masked store, and this implementation's was measured to cost an allocation per
 * stored vector. Those rows, at most [tileRows] minus one of them, are scalar, and [implementationsFor]
 * names that rather than letting the vector body's name stand for it.
 *
 * Resolving the species is what initializing this costs, so a runtime without the module must not reach it:
 * [com.eignex.koblas.BuiltinEngines] offers no engine holding this backend there. Where the module is
 * present but the object is asked for a product anyway, the portable tile is what runs, and
 * [implementationsFor] says so.
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
        // Spent once for the whole window, which leaves the tiles below with one writeback and no branch on
        // the multiplier. Three writeback forms meeting at one store is what makes a compiler stop keeping
        // the accumulators in registers, and this tile holds eight of them.
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
     * One whole tile, in as many accumulators as this runtime can hold at once.
     *
     * Where the machine has a fused multiply-add, a step of the depth is eight instructions into eight
     * accumulators and nothing else is live but the two loaded rows and the broadcast coefficient. Where it
     * does not, the same step is a multiply into a temporary and then an add, so it asks the compiler to
     * keep eight more vector values alive at once, and the compiler stops keeping them in registers: it
     * materialises some of them on the heap, which a probe sees as bytes per call and the arithmetic sees
     * as about half the throughput. Two of the four columns at a time fit either way, at the cost of
     * reading the left panel twice.
     *
     * So the decomposition follows the same gate the arithmetic does. This is not a change of geometry:
     * [tileRows] and [tileColumns] are what they were, the packed layout is unchanged, and each destination
     * entry still accumulates its own products in depth order, so both routes give the same answer bit for
     * bit as each other. What differs is how many of them are in flight.
     *
     * The stage evidence has both bodies measured at both widths. With the instruction, eight accumulators
     * lead the split by about 1.5x and neither allocates; without it, eight allocate and run at about half
     * the split's rate. One machine, and the choice here is between two bodies of one kernel rather than a
     * crossover between shapes.
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
     * Two columns of a tile in four accumulators, which is what fits where a step takes two instructions.
     *
     * The rows are both lane blocks, so each loaded left vector still serves both of this pair's columns,
     * and the padding of a group the destination does not fill is accumulated and then not stored, exactly
     * as the whole tile does.
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
     * The last row block of a destination whose rows do not fill a whole tile, in scalar arithmetic.
     *
     * The packed panel is padded and could be read by the vector body; what cannot be is the destination,
     * whose column ends part way through a lane block, and whose next column begins immediately after. The
     * Vector API offers a masked store for exactly this, and on this implementation it costs an allocation
     * per stored vector, which an allocation probe over a product with an edge shows and one without hides.
     * So the edge is scalar, it is at most [tileRows] minus one rows of the whole product, and
     * [implementationsFor] names it rather than letting the vector body's name stand for it.
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
     * `c += alpha · accumulated` over a whole lane block, one straight-line writeback and no branch.
     *
     * Inline, and measured to need it. A helper that takes a [DoubleVector] is as much of a risk as one that
     * returns one: an argument has to exist as a value at the call, so where the compiler declines to inline
     * the call the accumulators this tile holds in registers become heap objects. So does a writeback with a
     * branch in it, because the vectors the branches produce meet at the store; that is why the destination
     * multiplier is spent before the tiles rather than inside them. Both were measured here rather than
     * reasoned about: an uninstrumented probe over a whole product is what caught each of them, which is why
     * one runs over this tile, over a block of them and over the products around it.
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
