package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.internal.kernels.JvmCKernelBindings

/**
 * Whether the `jdk.incubator.vector` module resolved at runtime. The SIMD code lives in [SimdOps] so its
 * initializer only runs when this probe succeeds, and a JVM without the module takes the C path.
 */

internal val simdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.DoubleVector")
    true
} catch (_: Throwable) {
    false
}

internal val cKernelsAvailable: Boolean = !simdAvailable && JvmCKernelBindings.isAvailable

internal actual object PlatformKernels : Kernels, ArithmeticKernels {
    private val selected: Kernels = when {
        simdAvailable -> SimdKernels
        cKernelsAvailable -> CKernels
        else -> ScalarKernels
    }

    actual override val name: String get() = selected.name

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        selected.dot(a, aOff, b, bOff, len)

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        selected.axpy(y, yOff, alpha, x, xOff, len)

    actual override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        (selected as ArithmeticKernels).axpyArithmetic(y, yOff, alpha, x, xOff, len)

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = selected.scale(v, vOff, alpha, len)

    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = selected.nrm2(v, vOff, len)

    actual override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        selected.swap(a, aOff, b, bOff, len)

    actual override fun sum(v: DoubleArray, vOff: Int, len: Int): Double = selected.sum(v, vOff, len)

    actual override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        selected.ssqd(a, aOff, b, bOff, len)

    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = selected.asum(v, vOff, len)

    actual override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens =
        selected.rotmg(d1, d2, x1, y1)

    @Suppress("LongParameterList")
    actual override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: ModifiedGivens,
    ) = selected.rotm(x, xOff, xStride, y, yOff, yStride, len, transformation)

    @Suppress("LongParameterList")
    actual override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        selected.rot(x, xOff, y, yOff, len, c, s)

    @Suppress("LongParameterList") // four column offsets plus the shared operand
    actual override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) = selected.dot4(a, aOff, stride, b, bOff, len, out, outOff)

    @Suppress("LongParameterList")
    actual override fun axpy4(
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
    ) = selected.axpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)

    @Suppress("LongParameterList")
    actual override fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double = selected.dotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)

    @Suppress("LongParameterList")
    actual override fun trsmTile(
        validRows: Int,
        order: Int,
        packedTriangle: DoubleArray,
        triangleOff: Int,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOff: Int,
    ) = selected.trsmTile(validRows, order, packedTriangle, triangleOff, lower, unitDiag, x, xOff)

    @Suppress("LongParameterList")
    actual override fun gemmTrsmTile(
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
    ) = selected.gemmTrsmTile(
        depth, validRows, order, packedA, aOff, packedB, bOff,
        packedTriangle, triangleOff, lower, unitDiag, x, xOff,
    )

    override val gemmTileRows: Int get() = selected.gemmTileRows

    override val gemmTileCols: Int get() = selected.gemmTileCols

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
    ) = selected.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
}
