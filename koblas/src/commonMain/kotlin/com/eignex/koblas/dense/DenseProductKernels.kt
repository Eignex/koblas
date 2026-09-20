@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.dense

/**
 * The register-tile arithmetic a matrix product block is made of, with the tile geometry chosen locally.
 *
 * A block is a rectangle of the destination accumulated from two packed panels. What the caller states is
 * logical: how many rows, how many columns and how many steps of the shared dimension, plus where the
 * destination window is. What this backend owns is the register tile it cuts that rectangle into, which is
 * [tileRows] by [tileColumns] and is not a lane count, a panel execution group or a cache block. Those four
 * numbers are independent, and no algorithm above this contract writes one of them into its loops.
 *
 * The packed layout is the one [PackedLayout] describes, and the two are one contract: an operand is packed
 * into groups this backend's tile can consume along that operand's own axis. [PackedLayout.group] carries
 * that width and is the whole of what a consumer checks, so a left panel grouped by eight is readable by an
 * eight by four tile and by an eight by two one alike.
 *
 * Padding inside a packed group is positive zero, and what that promises is where the results go rather than
 * what they are: a padded zero against an infinite entry of the other operand evaluates to a NaN, and it
 * stays in a lane the destination does not have. A block writes exactly the `rows` by `columns` window it
 * was given and nothing beside it, which is what lets a caller select part of a destination.
 *
 * Scaling follows the BLAS convention and is applied once per block: `alpha` multiplies the product this
 * block accumulated, `beta` the destination window, and a zero `beta` overwrites without reading what is
 * there. A caller splitting the shared dimension into several depth blocks passes the real `beta` to the
 * first and `1.0` to every later one, which is how beta reaches an output exactly once however the depth
 * was cut.
 *
 * Implementations validate nothing and allocate nothing. The destination window must be disjoint from both
 * packed panels, which a packed panel being its own storage already gives.
 *
 * Results agree with the direct definition to within rounding rather than bit for bit: a tile accumulates in
 * whatever order and grouping its registers suit, and may fuse a multiply and an add.
 */
public interface DenseProductKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /**
     * Destination rows one register tile accumulates, which is the group a left panel is packed in.
     *
     * Positive and fixed for the life of an implementation, because a packed panel outlives the call that
     * produced it and has to stay readable by the same tile.
     */
    public val tileRows: Int

    /** Destination columns one register tile accumulates, which is the group a right panel is packed in. */
    public val tileColumns: Int

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
     * Building a list belongs where a route is built, before a timed region and never inside one.
     */
    public fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String>

    /**
     * Whether a product of these logical extents is worth packing into this backend's tiles at all.
     *
     * Packing costs a pass over both operands and buys contiguous, tile-shaped reads for the whole product.
     * The arithmetic grows with `rows · columns · depth` and the copy with `rows · depth + depth · columns`,
     * so a product with any small extent pays for a copy it cannot amortise, and a matrix-vector shaped call
     * pays for one it can never amortise at all. Where this answers false the caller runs the product as
     * panel work over the operands where they are, which is what avoids copying both of them into tiles. It
     * is not a promise that nothing is copied: that route may still gather one strided coefficient column,
     * borrow a destination column to accumulate into, and stage an operand that shares the destination.
     *
     * The crossover is local: a tile that finishes more arithmetic per loaded value pays off a copy sooner.
     */
    public fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean

    /**
     * `C = alpha · A · B + beta · C` over one `rows` by `columns` window, accumulating `depth` steps.
     *
     * [packedA] holds the left operand in [tileRows]-wide groups and [packedB] the right in
     * [tileColumns]-wide ones. Within a group, depth steps are adjacent blocks of that width, so entry
     * `(i, p)` of the left operand sits at `aOffset + (i / tileRows) · aGroupStride + p · tileRows +
     * i % tileRows`, and entry `(p, j)` of the right at `bOffset + (j / tileColumns) · bGroupStride +
     * p · tileColumns + j % tileColumns`.
     *
     * The group strides are the caller's because a panel may be a block packed for this call, where a group
     * is `depth · tileRows` long, or a retained panel covering a longer shared dimension, where the same
     * group is as long as that whole dimension and this block reads a slice of it. Passing the stride is
     * what makes one contract serve both without the panel having to be repacked.
     *
     * The destination window starts at [cOffset] with columns [ldc] apart. With no rows or no columns
     * nothing happens and nothing is read. With no depth the product is empty, so the window becomes
     * `beta · C` and neither packed panel is read.
     */
    public fun productBlock(
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
