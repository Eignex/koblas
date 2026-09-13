package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens

/** Shared vector dispatch; arithmetic stays in the independently callable implementations. */
internal class PolicyVectorKernels(
    private val runtime: DenseVectorKernels,
    private val native: DenseVectorKernels,
    private val dispatch: DenseDispatch,
) : DenseVectorKernels {
    override val name: String get() = "${runtime.name}+${native.name}-policy"

    private fun selected(operation: DenseOperation, length: Int): DenseVectorKernels =
        if (dispatch.usesNative(operation, length)) native else runtime

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        selected(DenseOperation.Dot, len).dot(a, aOff, b, bOff, len)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        selected(DenseOperation.Axpy, len).axpy(y, yOff, alpha, x, xOff, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) =
        selected(DenseOperation.Scale, len).scale(v, vOff, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        selected(DenseOperation.Nrm2, len).nrm2(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        selected(DenseOperation.Asum, len).asum(v, vOff, len)

    override fun iamax(v: DoubleArray, vOff: Int, len: Int): Int =
        selected(DenseOperation.Iamax, len).iamax(v, vOff, len)

    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens = runtime.rotmg(d1, d2, x1, y1)

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
    ) = selected(DenseOperation.Rotm, len).rotm(x, xOff, xStride, y, yOff, yStride, len, transformation)

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        selected(DenseOperation.Rot, len).rot(x, xOff, y, yOff, len, c, s)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        selected(DenseOperation.Swap, len).swap(a, aOff, b, bOff, len)

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double = selected(DenseOperation.Sum, len).sum(v, vOff, len)

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        selected(DenseOperation.Ssqd, len).ssqd(a, aOff, b, bOff, len)
}
