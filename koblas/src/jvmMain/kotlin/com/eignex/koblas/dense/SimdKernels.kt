package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.applyModifiedGivens
import com.eignex.koblas.internal.configuration.ImplementationNames
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
import com.eignex.koblas.portableRotmg
import kotlin.math.sqrt

/** The JVM Vector API kernels without automatic C selection. */
internal object SimdKernels : Kernels, ArithmeticKernels {
    private val lanes: Int = if (simdAvailable) SimdOps.lanes() else 0

    override val name: String get() = "${ImplementationNames.SIMD}($lanes lanes)"

    val isAvailable: Boolean get() = simdAvailable

    private fun vectorizes(len: Int): Boolean = simdAvailable && len >= lanes

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (vectorizes(len)) SimdOps.dot(a, aOff, b, bOff, len) else scalarDot(a, aOff, b, bOff, len)

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (vectorizes(len)) SimdOps.sum(v, vOff, len) else scalarSum(v, vOff, len)

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (vectorizes(len)) {
        SimdOps.ssqd(a, aOff, b, bOff, len)
    } else {
        scalarSsqd(a, aOff, b, bOff, len)
    }

    // No CBLAS or C routine generates the modified Givens transformation, so the portable one is the
    // implementation rather than a fallback.
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens = portableRotmg(d1, d2, x1, y1)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (vectorizes(len)) SimdOps.axpy(y, yOff, alpha, x, xOff, len) else scalarAxpy(y, yOff, alpha, x, xOff, len)
    }

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (vectorizes(len)) {
            SimdOps.axpyArithmetic(y, yOff, alpha, x, xOff, len)
        } else {
            scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (vectorizes(len)) SimdOps.scale(v, vOff, alpha, len) else scalarScale(v, vOff, alpha, len)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double {
        if (vectorizes(len)) {
            val squares = SimdOps.dot(v, vOff, v, vOff, len)
            if (squares.isFinite() && squares >= MIN_NORMAL) return sqrt(squares)
        }
        return euclideanNorm(v, vOff, len)
    }

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (vectorizes(len)) SimdOps.asum(v, vOff, len) else absoluteSum(v, vOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (vectorizes(len)) SimdOps.swap(a, aOff, b, bOff, len) else scalarSwap(a, aOff, b, bOff, len)
    }

    @Suppress("LongParameterList")
    override val gemmTileRows: Int get() = if (simdAvailable) SimdGemmTile.rows else super.gemmTileRows

    override val gemmTileCols: Int get() = if (simdAvailable) SimdGemmTile.COLUMNS else super.gemmTileCols

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
        if (simdAvailable) {
            SimdGemmTile.addProduct(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
        } else {
            super.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
        }
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
        if (!simdAvailable) {
            super.gemmTrsmTile(
                depth, validRows, order, packedA, aOff, packedB, bOff,
                packedTriangle, triangleOff, lower, unitDiag, x, xOff,
            )
            return
        }
        if (validRows == gemmTileRows && order == gemmTileCols) {
            SimdGemmTile.subtractProduct(depth, packedA, aOff, packedB, bOff, x, xOff)
        } else {
            SimdGemmTile.subtractProductEdge(
                depth, validRows, order, packedA, aOff, packedB, bOff, x, xOff,
            )
        }
        portableTrsmTile(
            gemmTileRows, gemmTileCols, validRows, order,
            packedTriangle, triangleOff, lower, unitDiag, x, xOff,
        )
    }

    override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        if (vectorizes(len)) {
            SimdOps.dot4(a, aOff, stride, b, bOff, len, out, outOff)
        } else {
            scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
        }
    }

    @Suppress("LongParameterList")
    override fun axpy4(
        y: DoubleArray,
        yOff: Int,
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        c0: Double,
        c1: Double,
        c2: Double,
        c3: Double,
        len: Int,
    ) {
        if (vectorizes(len)) {
            SimdOps.axpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
        } else {
            scalarAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
        }
    }

    @Suppress("LongParameterList")
    override fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double = if (vectorizes(len)) {
        SimdOps.dotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
    } else {
        scalarDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
    }

    @Suppress("LongParameterList")
    override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: ModifiedGivens,
    ) {
        if (transformation.flag == -2.0) return
        if (vectorizes(len) && xStride == 1 && yStride == 1) {
            SimdOps.rotm(
                x,
                xOff,
                y,
                yOff,
                len,
                transformation.h11,
                transformation.h12,
                transformation.h21,
                transformation.h22,
            )
        } else {
            applyModifiedGivens(x, xOff, xStride, y, yOff, yStride, len, transformation)
        }
    }

    // A plane rotation is the modified Givens transformation (c, s, -s, c), so it goes to the same kernel.
    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (vectorizes(len)) {
            SimdOps.rotm(x, xOff, y, yOff, len, c, s, -s, c)
        } else {
            portableRot(x, xOff, y, yOff, len, c, s)
        }
    }
}
