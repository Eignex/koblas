package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings

/** Bundled-C packed tiles with measured portable JVM fallbacks. */
internal class CPackedKernels(private val bindings: JvmCKernelBindings, private val exact: Boolean) : PackedKernels {
    private val gemmTileCCrossover = DenseTuning.jvmCGemmTileCrossover
    private val gemmTrsmTileCCrossover = DenseTuning.jvmCGemmTrsmTileCrossover

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
        if (!exact && depth < gemmTileCCrossover) {
            PortablePackedKernels.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
        } else {
            bindings.denseGemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
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
    ) {
        if (validRows == 0 || order == 0) return
        if (exact) {
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
        } else {
            PortablePackedKernels.trsmTile(validRows, order, packedTriangle, triangleOff, lower, unitDiag, x, xOff)
        }
    }

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
        if (!exact && depth < gemmTrsmTileCCrossover) {
            PortablePackedKernels.gemmTrsmTile(
                depth, validRows, order, packedA, aOff, packedB, bOff,
                packedTriangle, triangleOff, lower, unitDiag, x, xOff,
            )
            return
        }
        bindings.denseGemmTrsmTile(
            depth, validRows, order, packedA, aOff, packedB, bOff,
            packedTriangle, triangleOff, if (lower) 1 else 0, if (unitDiag) 1 else 0, x, xOff,
        )
    }
}
