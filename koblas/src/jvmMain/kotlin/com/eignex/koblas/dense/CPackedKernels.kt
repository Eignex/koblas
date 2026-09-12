package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings

/** Exact native packed leaves; performance policy is resolved before entering this implementation. */
internal class CPackedKernels(private val bindings: JvmCKernelBindings) : PackedKernels {
    override val gemmTileRows: Int get() = PORTABLE_TILE
    override val gemmTileCols: Int get() = PORTABLE_TILE

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
    ) {
        if (depth == 0) return
        bindings.denseGemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
    }

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
    ) {
        if (validRows == 0 || order == 0) return
        bindings.denseTrsmTile(
            validRows,
            order,
            packedTriangle,
            triangleOff,
            if (lower) 1 else 0,
            if (unitDiag) 1 else 0,
            x,
            xOff,
        )
    }

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
    ) {
        if (validRows == 0 || order == 0) return
        bindings.denseGemmTrsmTile(
            depth,
            validRows,
            order,
            packedA,
            aOff,
            packedB,
            bOff,
            packedTriangle,
            triangleOff,
            if (lower) 1 else 0,
            if (unitDiag) 1 else 0,
            x,
            xOff,
        )
    }
}
