package com.eignex.koblas.dense

/** Arithmetic over explicitly padded packed panels and their fixed platform tile shape. */
public interface PackedKernels {
    /** Physical row count of a packed output tile. */
    public val gemmTileRows: Int

    /** Physical column count of a packed output tile. */
    public val gemmTileCols: Int

    /** Adds a product of two padded panels into one full output tile. */
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

    /** Solves `X * T = B` in place for the logical edge of one packed tile. */
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

    /** Subtracts one packed product from [x], then solves the logical tile through the packed triangle. */
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
