package com.eignex.koblas.dense

/**
 * Arithmetic over explicitly padded packed panels and one fixed platform tile shape. Callers own validation:
 * dimensions must be non-negative, offsets and physical panel windows must be valid, and input panels must not
 * overlap destinations. Short logical edges occupy the same physical shape and use positive-zero padding.
 */
public interface PackedKernels {
    /**
     * Positive physical row count of a packed output tile. It is fixed for the life of an implementation because
     * callers use it when allocating and packing panels.
     */
    public val gemmTileRows: Int

    /**
     * Positive physical column count of a packed output tile. It is fixed for the life of an implementation.
     */
    public val gemmTileCols: Int

    /**
     * Adds one packed product into a full [gemmTileRows] by [gemmTileCols] output tile. [packedA] contains
     * [depth] groups of [gemmTileRows] contiguous values beginning at [aOff], and [packedB] contains [depth]
     * groups of [gemmTileCols] contiguous values beginning at [bOff]. Output columns begin at [cOff] and are
     * [ldc] elements apart. Every group is physically full; short edges must be zero padded by the caller.
     * A zero [depth] leaves the output unchanged without reading either packed panel.
     */
    @Suppress("LongParameterList")
    public fun gemmTile(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
    )

    /**
     * Solves `X * T = B` in place for one logical tile edge. [packedTriangle] stores `T` with physical row
     * stride [gemmTileCols], while [x] stores columns with physical stride [gemmTileRows]. [validRows] must be
     * in `0..gemmTileRows` and [order] in `0..gemmTileCols`. Only those logical rows and columns of [x] are
     * overwritten; its padding is preserved. A zero logical dimension performs no reads. When [unitDiag] is
     * true, diagonal entries of [packedTriangle] are not read. [lower] selects the stored triangle of `T`.
     */
    @Suppress("LongParameterList")
    public fun trsmTile(
        validRows: Int,
        order: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOff: Int,
    )

    /**
     * Subtracts one packed product from the logical [validRows] by [order] edge of [x], then solves that edge
     * through [packedTriangle] using the [trsmTile] layout and triangle rules. [packedA] and [packedB] use the
     * [gemmTile] depth-group layouts. Entries of [x] outside the logical edge are preserved. A zero logical
     * dimension performs no reads; a zero [depth] performs only the triangular solve.
     */
    @Suppress("LongParameterList")
    public fun gemmTrsmTile(
        depth: Int,
        validRows: Int,
        order: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOff: Int,
    )
}

/** Portable packed layout shape and tile arithmetic used by the scalar engine. */
internal object PortablePackedKernels : PackedKernels {
    override val gemmTileRows: Int get() = PORTABLE_TILE
    override val gemmTileCols: Int get() = PORTABLE_TILE

    override fun gemmTile(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
    ) = portableGemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)

    override fun trsmTile(
        validRows: Int,
        order: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOff: Int,
    ) = portableTrsmTile(
        gemmTileRows, gemmTileCols, validRows, order,
        packedTriangle, triangleOff, lower, unitDiag, x, xOff,
    )

    override fun gemmTrsmTile(
        depth: Int,
        validRows: Int,
        order: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOff: Int,
    ) = portableGemmTrsmTile(
        gemmTileRows, gemmTileCols, depth, validRows, order,
        packedA, aOff, packedB, bOff, packedTriangle, triangleOff,
        lower, unitDiag, x, xOff,
    )
}
