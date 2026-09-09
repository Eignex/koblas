package com.eignex.koblas.dense

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
