package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.numeric.MIN_NORMAL
import com.eignex.koblas.internal.numeric.rescaledEuclideanNorm
import kotlin.math.sqrt

/** Check the optional module without initializing a Vector API implementation. */
internal val simdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.DoubleVector")
    true
} catch (_: ClassNotFoundException) {
    false
}

/** Only supplied operations are capabilities; the shared interface owns the portable defaults. */
private object SimdVectorLeaves : KoblasEngine() {
    override val id: String = "jvm-simd"
    override val implementations = setOf(Dot, Sum, Nrm2, Iamax, Asum, Axpy)
    override val dynamicNorm: Boolean get() = true
    override val contiguousVectors: Boolean get() = true
    override val minimumVectorLength: Int get() = SimdOps.lanes()

    override val name: String = "simd(${SimdOps.lanes()} lanes)"
    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = SimdOps.dot(a, aOff, b, bOff, len)
    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = SimdOps.sum(v, vOff, len)
    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = SimdOps.asum(v, vOff, len)
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int = SimdOps.iamax(v, vOff, len)
    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        val squares = SimdOps.dot(v, vOff, v, vOff, len)
        return if (squares.isFinite() && squares >= MIN_NORMAL) {
            sqrt(squares)
        } else {
            rescaledEuclideanNorm(v, vOff, vStride, len, squares)
        }
    }

    @Suppress("LongParameterList")
    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) = SimdOps.axpy(y, yOff, alpha, x, xOff, len)
}

internal val simdContributions: List<KoblasEngine> by lazy {
    listOf(SimdVectorLeaves, SparseSimd, SimdPanelKernels, SimdProductKernels, SimdTriangularKernels)
}
