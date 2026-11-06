package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.engine.EngineImplementation.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.kernels.scalar.dense.*
import kotlin.math.sqrt

/** Check the optional module without initializing a Vector API implementation. */
internal val simdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.DoubleVector")
    true
} catch (_: ClassNotFoundException) {
    false
}

/** Only supplied operations are capabilities; the composer supplies other providers. */
private object SimdVectorLeaves : EngineKernels {
    override val id: String = "jvm-simd"
    override val implementations = setOf(Dot, Sum, Nrm2, Iamax, Asum, Axpy)
    override fun dynamicAttribution(implementation: EngineImplementation): Boolean = implementation == Nrm2
    override fun requiresUnitStride(implementation: EngineImplementation): Boolean = true
    override fun minimumLength(implementation: EngineImplementation): Int = SimdOps.lanes()

    override val name: String = "simd(${SimdOps.lanes()} lanes)"
    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double {
        require(aStride == 1 && bStride == 1)
        return SimdOps.dot(a, aOff, b, bOff, len)
    }
    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        require(vStride == 1)
        return SimdOps.sum(v, vOff, len)
    }
    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        require(vStride == 1)
        return SimdOps.asum(v, vOff, len)
    }
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        require(vStride == 1)
        return SimdOps.iamax(v, vOff, len)
    }
    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        require(vStride == 1)
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
    ) {
        require(yStride == 1 && xStride == 1)
        SimdOps.axpy(y, yOff, alpha, x, xOff, len)
    }
}

internal val simdContributions: List<EngineKernels> by lazy {
    listOf(SimdVectorLeaves, SparseSimd, SimdPanelKernels, SimdProductKernels, SimdTriangularKernels)
}
