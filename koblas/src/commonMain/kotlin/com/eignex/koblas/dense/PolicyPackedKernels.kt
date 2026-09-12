package com.eignex.koblas.dense

/** Shared packed dispatch; arithmetic stays in the independently callable implementations. */
internal class PolicyPackedKernels(
    private val runtime: PackedKernels,
    private val native: PackedKernels,
    private val dispatch: DenseDispatch,
) : PackedKernels {
    override val gemmTileRows: Int get() = runtime.gemmTileRows
    override val gemmTileCols: Int get() = runtime.gemmTileCols

    private fun selected(operation: DenseOperation, length: Int): PackedKernels =
        if (dispatch.usesNative(operation, length)) native else runtime

    @Suppress("LongParameterList")
    override fun gemmTile(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
    ) = selected(DenseOperation.GemmTile, depth).gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)

    @Suppress("LongParameterList")
    override fun trsmTile(
        validRows: Int,
        order: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOff: Int,
    ) = selected(DenseOperation.TrsmTile, order).trsmTile(
        validRows, order, packedTriangle, triangleOff, lower, unitDiag, x, xOff,
    )

    @Suppress("LongParameterList")
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
    ) = selected(DenseOperation.GemmTrsmTile, depth).gemmTrsmTile(
        depth, validRows, order, packedA, aOff, packedB, bOff, packedTriangle, triangleOff, lower, unitDiag, x, xOff,
    )
}
