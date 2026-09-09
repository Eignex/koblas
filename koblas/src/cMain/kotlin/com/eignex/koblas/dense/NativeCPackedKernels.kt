@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.koblas_dense_gemm_tile
import com.eignex.koblas.internal.kernels.koblas_dense_gemm_trsm_tile
import com.eignex.koblas.internal.kernels.koblas_dense_trsm_tile
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** Native compiled-C packed tile arithmetic. */
internal object NativeCPackedKernels : PackedKernels {
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
    ) {
        if (depth == 0) return
        packedA.usePinned { ap ->
            packedB.usePinned { bp ->
                c.usePinned { cp ->
                    koblas_dense_gemm_tile(
                        depth, ap.addressOf(0), aOff, bp.addressOf(0), bOff, cp.addressOf(0), cOff, ldc,
                    )
                }
            }
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
        packedTriangle.usePinned { triangle ->
            x.usePinned { result ->
                koblas_dense_trsm_tile(
                    validRows, order, triangle.addressOf(0), triangleOff,
                    if (lower) 1 else 0, if (unitDiag) 1 else 0, result.addressOf(0), xOff,
                )
            }
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
        if (depth == 0) return trsmTile(validRows, order, packedTriangle, triangleOff, lower, unitDiag, x, xOff)
        packedA.usePinned { left ->
            packedB.usePinned { right ->
                packedTriangle.usePinned { triangle ->
                    x.usePinned { result ->
                        koblas_dense_gemm_trsm_tile(
                            depth, validRows, order, left.addressOf(0), aOff, right.addressOf(0), bOff,
                            triangle.addressOf(0), triangleOff, if (lower) 1 else 0, if (unitDiag) 1 else 0,
                            result.addressOf(0), xOff,
                        )
                    }
                }
            }
        }
    }
}
