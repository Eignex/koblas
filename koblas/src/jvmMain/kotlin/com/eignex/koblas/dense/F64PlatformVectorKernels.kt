package com.eignex.koblas.dense

import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.numeric.F64_MIN_NORMAL
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.internal.numeric.scalarAxpy
import com.eignex.koblas.internal.numeric.scalarDot
import com.eignex.koblas.internal.numeric.scalarDot4
import com.eignex.koblas.internal.numeric.scalarScale
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators
import kotlin.math.sqrt

/**
 * Whether the `jdk.incubator.vector` module resolved at runtime. The SIMD code lives in [Simd] so its
 * initializer only runs when this probe succeeds, and a JVM without the module takes the scalar path.
 */

internal val simdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.DoubleVector")
    true
} catch (_: Throwable) {
    false
}

/**
 * Lane width of the vector path, or 0 when it is unavailable. A run below one lane executes no vector
 * body at all, so short runs take the scalar kernels rather than pay the vector prologue.
 */
private val simdLanes: Int = if (simdAvailable) Simd.lanes() else 0

internal actual object F64PlatformVectorKernels : F64VectorKernels {
    actual override val name: String
        get() = if (simdAvailable) "${BackendNames.SIMD}($simdLanes lanes)" else BackendNames.SCALAR

    override val isPortable: Boolean get() = true

    /**
     * Always true: without the vector module these kernels run the scalar loops rather than nothing, so the
     * SIMD probe shows up in [name] instead of here.
     */
    override val isAvailable: Boolean get() = true

    private fun vectorizes(len: Int): Boolean = simdAvailable && len >= simdLanes

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (vectorizes(len)) Simd.dot(a, aOff, b, bOff, len) else scalarDot(a, aOff, b, bOff, len)

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (vectorizes(len)) {
            Simd.axpy(y, yOff, alpha, x, xOff, len)
        } else {
            scalarAxpy(y, yOff, alpha, x, xOff, len)
        }
    }

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (vectorizes(len)) {
            Simd.scale(v, vOff, alpha, len)
        } else {
            scalarScale(v, vOff, alpha, len)
        }
    }

    /**
     * The plain sum of squares vectorizes; only the rescaling that overflow or underflow forces does not,
     * and that path defers to the shared implementation.
     */
    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double {
        if (vectorizes(len)) {
            val squares = Simd.dot(v, vOff, v, vOff, len)
            if (squares.isFinite() && squares >= F64_MIN_NORMAL) return sqrt(squares)
        }
        return euclideanNorm(v, vOff, len)
    }

    /** No SIMD counterpart worth writing, so the shared common implementation serves the JVM too. */
    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)

    /** Overridden because [Simd.dot4] loads each b segment once for all four columns. */
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
    ) {
        if (vectorizes(len)) {
            Simd.dot4(a, aOff, stride, b, bOff, len, out, outOff)
        } else {
            scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
        }
    }
}

internal object Simd {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = SPECIES.length()

    fun lanes(): Int = LANE

    /**
     * Run length from which four accumulators beat one, as a multiple of the lane width because the reduce
     * that combines them is a fixed cost against a per-vector saving.
     *
     * Set where `jvmLevel1Benchmark` shows the win unambiguously rather than at the crossover: on four lanes
     * the unrolled form runs 2.2x to 2.8x at 1024 and 4096, and short runs there measure too noisily to site
     * a threshold on. Below this the run takes [dotOneChain], which is the loop this kernel had before the
     * unroll existed, so nothing short can regress.
     */
    private val UNROLL_MIN = 32 * LANE

    /**
     * One accumulator chains every multiply-add on the previous one's result, and an FMA's latency is
     * several times its throughput, so a single chain leaves most of the unit idle on a long run. Four
     * independent chains keep it fed, which is why [dot4] already runs that many. Short runs stay on the
     * single chain: there the three vector adds that combine the four cost more than the chain they break,
     * and unrolling them unconditionally measured slower than not unrolling at all.
     */
    fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (len >= UNROLL_MIN) dotUnrolled(a, aOff, b, bOff, len) else dotOneChain(a, aOff, b, bOff, len)

    private fun dotOneChain(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        var i = 0
        val bound = SPECIES.loopBound(len)
        var sum = DoubleVector.zero(SPECIES)
        while (i < bound) {
            val va = DoubleVector.fromArray(SPECIES, a, aOff + i)
            val vb = DoubleVector.fromArray(SPECIES, b, bOff + i)
            sum = va.fma(vb, sum)
            i += LANE
        }
        var s = sum.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            s += a[aOff + i] * b[bOff + i]
            i++
        }
        return s
    }

    /** [dot] past [UNROLL_MIN], where the four chains pay for the reduce that combines them. */
    private fun dotUnrolled(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        var i = 0
        val unrolled = len - len % (4 * LANE)
        while (i < unrolled) {
            s0 = DoubleVector.fromArray(SPECIES, a, aOff + i)
                .fma(DoubleVector.fromArray(SPECIES, b, bOff + i), s0)
            s1 = DoubleVector.fromArray(SPECIES, a, aOff + i + LANE)
                .fma(DoubleVector.fromArray(SPECIES, b, bOff + i + LANE), s1)
            s2 = DoubleVector.fromArray(SPECIES, a, aOff + i + 2 * LANE)
                .fma(DoubleVector.fromArray(SPECIES, b, bOff + i + 2 * LANE), s2)
            s3 = DoubleVector.fromArray(SPECIES, a, aOff + i + 3 * LANE)
                .fma(DoubleVector.fromArray(SPECIES, b, bOff + i + 3 * LANE), s3)
            i += 4 * LANE
        }
        // The whole lanes the unroll left over, then the scalar tail below one lane.
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            s0 = DoubleVector.fromArray(SPECIES, a, aOff + i)
                .fma(DoubleVector.fromArray(SPECIES, b, bOff + i), s0)
            i += LANE
        }
        var s = s0.add(s1).add(s2).add(s3).reduceLanes(VectorOperators.ADD)
        while (i < len) {
            s += a[aOff + i] * b[bOff + i]
            i++
        }
        return s
    }

    /**
     * Four rows against one shared vector, each b segment loaded once into four independent
     * accumulators. [F64LinearAlgebra.gemv] wants this over four [dot] calls and their four reductions.
     */
    @Suppress("LongParameterList") // four row offsets plus the shared operand
    fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        val o1 = aOff + stride
        val o2 = aOff + 2 * stride
        val o3 = aOff + 3 * stride
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val vb = DoubleVector.fromArray(SPECIES, b, bOff + i)
            s0 = DoubleVector.fromArray(SPECIES, a, aOff + i).fma(vb, s0)
            s1 = DoubleVector.fromArray(SPECIES, a, o1 + i).fma(vb, s1)
            s2 = DoubleVector.fromArray(SPECIES, a, o2 + i).fma(vb, s2)
            s3 = DoubleVector.fromArray(SPECIES, a, o3 + i).fma(vb, s3)
            i += LANE
        }
        var r0 = s0.reduceLanes(VectorOperators.ADD)
        var r1 = s1.reduceLanes(VectorOperators.ADD)
        var r2 = s2.reduceLanes(VectorOperators.ADD)
        var r3 = s3.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            val bi = b[bOff + i]
            r0 += a[aOff + i] * bi
            r1 += a[o1 + i] * bi
            r2 += a[o2 + i] * bi
            r3 += a[o3 + i] * bi
            i++
        }
        out[outOff] = r0
        out[outOff + 1] = r1
        out[outOff + 2] = r2
        out[outOff + 3] = r3
    }

    fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        val alphaVec = DoubleVector.broadcast(SPECIES, alpha)
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val vx = DoubleVector.fromArray(SPECIES, x, xOff + i)
            val vy = DoubleVector.fromArray(SPECIES, y, yOff + i)
            // y_new = alpha * x + y  ->  vx.fma(alphaVec, vy) computes vx * alphaVec + vy.
            vx.fma(alphaVec, vy).intoArray(y, yOff + i)
            i += LANE
        }
        while (i < len) {
            y[yOff + i] += alpha * x[xOff + i]
            i++
        }
    }

    fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (alpha == 1.0) return
        val alphaVec = DoubleVector.broadcast(SPECIES, alpha)
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val vv = DoubleVector.fromArray(SPECIES, v, vOff + i)
            vv.mul(alphaVec).intoArray(v, vOff + i)
            i += LANE
        }
        while (i < len) {
            v[vOff + i] *= alpha
            i++
        }
    }
}
