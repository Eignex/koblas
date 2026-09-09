package com.eignex.koblas.dense

/** JVM Vector API packed tile arithmetic. */
internal object SimdPackedKernels : PackedKernels {
    override val gemmTileRows: Int
        get() = if (simdAvailable) SimdGemmTile.rows else PortablePackedKernels.gemmTileRows

    override val gemmTileCols: Int
        get() = if (simdAvailable) SimdGemmTile.COLUMNS else PortablePackedKernels.gemmTileCols

    override fun gemmTile(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
    ) {
        if (simdAvailable) {
            SimdGemmTile.addProduct(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
        } else {
            PortablePackedKernels.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
        }
    }

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
    ) {
        if (validRows == 0 || order == 0) return
        if (!simdAvailable) {
            PortablePackedKernels.gemmTrsmTile(
                depth, validRows, order, packedA, aOff, packedB, bOff,
                packedTriangle, triangleOff, lower, unitDiag, x, xOff,
            )
            return
        }
        if (validRows == gemmTileRows && order == gemmTileCols) {
            SimdGemmTile.subtractProduct(depth, packedA, aOff, packedB, bOff, x, xOff)
        } else {
            SimdGemmTile.subtractProductEdge(depth, validRows, order, packedA, aOff, packedB, bOff, x, xOff)
        }
        portableTrsmTile(
            gemmTileRows, gemmTileCols, validRows, order,
            packedTriangle, triangleOff, lower, unitDiag, x, xOff,
        )
    }
}
