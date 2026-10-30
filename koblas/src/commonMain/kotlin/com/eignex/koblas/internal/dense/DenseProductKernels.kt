@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.internal.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/**
 * The register-tile arithmetic a matrix product block is made of, with the tile geometry chosen locally.
 *
 * A block is a rectangle of the destination accumulated from two packed panels. What the caller states is
 * logical: how many rows, how many columns and how many steps of the shared dimension, plus where the
 * destination window is. What this backend owns is the register tile it cuts that rectangle into, which is
 * [tileRows] by [tileColumns] and is not a lane count, a panel execution group or a cache block. Those four
 * numbers are independent, and no algorithm above this contract writes one of them into its loops.
 *
 * Transient panels group rows of the left operand and columns of the right by the corresponding tile
 * dimension, with contiguous depth steps inside each group.
 *
 * Padding inside a packed group is positive zero. It introduces no logical destination entries: a block
 * writes exactly the `rows` by `columns` window it was given and nothing beside it. Padding must not affect
 * entries inside that window.
 *
 * A block computes `C = alpha · A · B + beta · C`, allowing implementation-dependent scaling placement
 * and reduction order. A zero `beta` overwrites without reading the destination. A caller splitting the
 * shared dimension into several depth blocks passes the real `beta` to the
 * first and `1.0` to every later one, which is how beta reaches an output exactly once however the depth
 * was cut.
 *
 * Implementations validate nothing and allocate nothing. The destination window must be disjoint from both
 * packed panels, which a packed panel being its own storage already gives.
 *
 * Results agree with the direct definition to within rounding rather than bit for bit: a tile accumulates in
 * whatever order and grouping its registers suit, and may fuse a multiply and an add.
 */
internal interface DenseProductKernels {
    /** Short implementation identifier for diagnostics. */
    val name: String

    /**
     * Destination rows one register tile accumulates, which is the group a left panel is packed in.
     *
     * Positive and fixed for the life of an implementation.
     */
    val tileRows: Int

    /** Destination columns one register tile accumulates, which is the group a right panel is packed in. */
    val tileColumns: Int

    /**
     * Every arithmetic body a block of these extents reaches, in the order it reaches them.
     *
     * A list rather than a name, because a block is not one tile: it is cut into tiles of [tileRows] by
     * [tileColumns] plus whichever edges its extents leave, and those need not reach the same body. A
     * destination whose rows stop part way through a lane block is a different writeback from one whose
     * rows fill it, so a block that fills its tiles exactly answers with one entry and one with a remainder
     * answers with what the remainder reaches as well.
     *
     * The extents are a block's, which is what [productBlock] is called with; a caller describing a whole
     * product asks this of each block its schedule cuts. A block with no extent runs nothing and answers
     * with nothing. [name] identifies a selection instead, which is a different question: a backend whose
     * tile needs a vector unit answers here with the portable body where it has none.
     *
     * Building a list allocates, so it belongs before a timed region and never inside one.
     */
    fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String>

    /**
     * Whether a product of these logical extents is worth packing into this backend's tiles at all.
     *
     * Packing costs a pass over both operands and buys contiguous, tile-shaped reads for the whole product.
     * The arithmetic grows with `rows · columns · depth` and the copy with `rows · depth + depth · columns`,
     * so a product with any small extent pays for a copy it cannot amortise, and a matrix-vector shaped call
     * pays for one it can never amortise at all. Where this answers false the caller runs the product as
     * panel work over the operands where they are, which is what avoids copying both of them into tiles. It
     * is not a promise that nothing is copied: that path may still gather one strided coefficient column,
     * borrow a destination column to accumulate into, and stage an operand that shares the destination.
     *
     * The crossover is local: a tile that finishes more arithmetic per loaded value pays off a copy sooner.
     */
    fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean

    /**
     * `C = alpha · A · B + beta · C` over one `rows` by `columns` window, accumulating `depth` steps.
     *
     * [packedA] holds the left operand in [tileRows]-wide groups and [packedB] the right in
     * [tileColumns]-wide ones. Within a group, depth steps are adjacent blocks of that width, so entry
     * `(i, p)` of the left operand sits at `aOffset + (i / tileRows) · aGroupStride + p · tileRows +
     * i % tileRows`, and entry `(p, j)` of the right at `bOffset + (j / tileColumns) · bGroupStride +
     * p · tileColumns + j % tileColumns`.
     *
     * The group strides are `depth · tileRows` and `depth · tileColumns` for transient panels.
     *
     * The destination window starts at [cOffset] with columns [ldc] apart. With no rows or no columns
     * nothing happens and nothing is read. With no depth the product is empty, so the window becomes
     * `beta · C` and neither packed panel is read.
     */
    fun productBlock(
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
    )
}
