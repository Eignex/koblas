package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings

/** Bundled-C packed tiles with measured portable JVM fallbacks. */
internal object CPackedKernels : PackedKernels {
    private val GEMM_TILE_C_CROSSOVER = DenseTuning.jvmCGemmTileCrossover
    private val GEMM_TRSM_TILE_C_CROSSOVER = DenseTuning.jvmCGemmTrsmTileCrossover

    override val gemmTileRows: Int get() = PortablePackedKernels.gemmTileRows
    override val gemmTileCols: Int get() = PortablePackedKernels.gemmTileCols

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
        if (depth < GEMM_TILE_C_CROSSOVER) {
            PortablePackedKernels.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
        } else {
            JvmCKernelBindings.denseGemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
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
    ) = PortablePackedKernels.trsmTile(
        validRows,
        order,
        packedTriangle,
        triangleOff,
        lower,
        unitDiag,
        x,
        xOff,
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
        if (depth < GEMM_TRSM_TILE_C_CROSSOVER) {
            PortablePackedKernels.gemmTrsmTile(
                depth, validRows, order, packedA, aOff, packedB, bOff,
                packedTriangle, triangleOff, lower, unitDiag, x, xOff,
            )
            return
        }
        JvmCKernelBindings.denseGemmTrsmTile(
            depth, validRows, order, packedA, aOff, packedB, bOff,
            packedTriangle, triangleOff, lower, unitDiag, x, xOff,
        )
    }
}
