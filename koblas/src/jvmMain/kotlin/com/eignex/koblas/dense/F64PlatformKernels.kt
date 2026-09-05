package com.eignex.koblas.dense

import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators
import kotlin.math.abs

/**
 * Whether the `jdk.incubator.vector` module resolved at runtime. The SIMD code lives in [Simd] so its
 * initializer only runs when this probe succeeds, and a JVM without the module takes the C path.
 */

internal val simdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.DoubleVector")
    true
} catch (_: Throwable) {
    false
}

internal val cKernelsAvailable: Boolean = !simdAvailable && JvmCKernelBindings.isAvailable

internal actual object F64PlatformKernels : F64Kernels, F64ArithmeticKernels {
    private val selected: F64Kernels = when {
        simdAvailable -> F64SimdKernels
        cKernelsAvailable -> F64CKernels
        else -> F64ScalarKernels
    }

    actual override val name: String get() = selected.name

    override val isPortable: Boolean get() = true

    /**
     * Always true: without the vector module these kernels run the bundled C implementation, so the
     * SIMD probe shows up in [name] instead of here.
     */
    override val isAvailable: Boolean get() = true

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        selected.dot(a, aOff, b, bOff, len)

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        selected.axpy(y, yOff, alpha, x, xOff, len)

    actual override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        (selected as F64ArithmeticKernels).axpyArithmetic(y, yOff, alpha, x, xOff, len)

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = selected.scale(v, vOff, alpha, len)

    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = selected.nrm2(v, vOff, len)

    actual override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        selected.swap(a, aOff, b, bOff, len)

    actual override fun sum(v: DoubleArray, vOff: Int, len: Int): Double = selected.sum(v, vOff, len)

    actual override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        selected.ssqd(a, aOff, b, bOff, len)

    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = selected.asum(v, vOff, len)

    actual override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
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
        transformation: F64ModifiedGivens,
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
}

internal object Simd {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = SPECIES.length()

    fun lanes(): Int = LANE

    /**
     * Run length from which four accumulators beat one, as a multiple of the lane width because the reduce
     * that combines them is a fixed cost against a per-vector saving.
     *
     * Set where `jvmLevel1Benchmark` shows the win without argument rather than at the crossover: on four
     * lanes the unrolled form runs about 2x to 3x at 1024 and 4096, while unrolling every length is slower
     * than not unrolling below 32 elements. The `LANE` scaling is reasoning, not measurement: only four
     * lanes were measured.
     */
    private val UNROLL_MIN = 32 * LANE

    /**
     * One accumulator chains every multiply-add on the previous one's result, and an FMA's latency is
     * several times its throughput, so a single chain leaves most of the unit idle on a long run. Four
     * independent chains keep it fed, which is why [dot4] already runs that many.
     *
     * Two functions rather than one branching body so the short-length arm stays small enough for the JIT
     * to inline into its callers, which is what the four accumulators and the extra loop would cost it.
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
        // What the unroll leaves over is under one unroll width, which is what the single chain is for.
        val head = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD)
        return head + dotOneChain(a, aOff + unrolled, b, bOff + unrolled, len - unrolled)
    }

    /**
     * One vector accumulator and a scalar tail. Deliberately not the four-chain form [dot] takes above
     * [UNROLL_MIN]: that threshold was measured for dot, and nothing here has measured where a sum's own
     * crossover sits.
     */
    fun sum(v: DoubleArray, vOff: Int, len: Int): Double {
        var i = 0
        val bound = SPECIES.loopBound(len)
        var total = DoubleVector.zero(SPECIES)
        while (i < bound) {
            total = total.add(DoubleVector.fromArray(SPECIES, v, vOff + i))
            i += LANE
        }
        var s = total.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            s += v[vOff + i]
            i++
        }
        return s
    }

    /**
     * [dot]'s load pattern with a subtract fused in, so it inherits the same accumulator reasoning and the
     * same [UNROLL_MIN]. That threshold is [dot]'s measurement adopted by analogy, not one of its own: the
     * loop differs from dot's by one vector subtract against identical loads.
     */
    fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (len >= UNROLL_MIN) {
        ssqdUnrolled(a, aOff, b, bOff, len)
    } else {
        ssqdOneChain(a, aOff, b, bOff, len)
    }

    private fun ssqdOneChain(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        var i = 0
        val bound = SPECIES.loopBound(len)
        var sum = DoubleVector.zero(SPECIES)
        while (i < bound) {
            val d = DoubleVector.fromArray(SPECIES, a, aOff + i).sub(DoubleVector.fromArray(SPECIES, b, bOff + i))
            sum = d.fma(d, sum)
            i += LANE
        }
        var s = sum.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            val d = a[aOff + i] - b[bOff + i]
            s += d * d
            i++
        }
        return s
    }

    /** [ssqd] past [UNROLL_MIN], where the four chains pay for the reduce that combines them. */
    private fun ssqdUnrolled(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        var i = 0
        val unrolled = len - len % (4 * LANE)
        while (i < unrolled) {
            val d0 = DoubleVector.fromArray(SPECIES, a, aOff + i).sub(DoubleVector.fromArray(SPECIES, b, bOff + i))
            s0 = d0.fma(d0, s0)
            val d1 = DoubleVector.fromArray(SPECIES, a, aOff + i + LANE)
                .sub(DoubleVector.fromArray(SPECIES, b, bOff + i + LANE))
            s1 = d1.fma(d1, s1)
            val d2 = DoubleVector.fromArray(SPECIES, a, aOff + i + 2 * LANE)
                .sub(DoubleVector.fromArray(SPECIES, b, bOff + i + 2 * LANE))
            s2 = d2.fma(d2, s2)
            val d3 = DoubleVector.fromArray(SPECIES, a, aOff + i + 3 * LANE)
                .sub(DoubleVector.fromArray(SPECIES, b, bOff + i + 3 * LANE))
            s3 = d3.fma(d3, s3)
            i += 4 * LANE
        }
        val head = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD)
        return head + ssqdOneChain(a, aOff + unrolled, b, bOff + unrolled, len - unrolled)
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

    /**
     * Absolute values summed. Vectorized because the JIT will not do it: splitting a sum across lanes
     * reorders the additions, which is a different result in floating point, so HotSpot leaves an FP-add
     * reduction alone however hot it gets. Four accumulators for the reason [dot] gives.
     */
    fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len >= UNROLL_MIN) asumUnrolled(v, vOff, len) else asumOneChain(v, vOff, len)

    private fun asumOneChain(v: DoubleArray, vOff: Int, len: Int): Double {
        var i = 0
        val bound = SPECIES.loopBound(len)
        var sum = DoubleVector.zero(SPECIES)
        while (i < bound) {
            sum = sum.add(DoubleVector.fromArray(SPECIES, v, vOff + i).abs())
            i += LANE
        }
        var s = sum.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            s += abs(v[vOff + i])
            i++
        }
        return s
    }

    /** [asum] past [UNROLL_MIN], where the four chains pay for the reduce that combines them. */
    private fun asumUnrolled(v: DoubleArray, vOff: Int, len: Int): Double {
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        var i = 0
        val unrolled = len - len % (4 * LANE)
        while (i < unrolled) {
            s0 = s0.add(DoubleVector.fromArray(SPECIES, v, vOff + i).abs())
            s1 = s1.add(DoubleVector.fromArray(SPECIES, v, vOff + i + LANE).abs())
            s2 = s2.add(DoubleVector.fromArray(SPECIES, v, vOff + i + 2 * LANE).abs())
            s3 = s3.add(DoubleVector.fromArray(SPECIES, v, vOff + i + 3 * LANE).abs())
            i += 4 * LANE
        }
        // What the unroll leaves over is under one unroll width, which is what the single chain is for.
        val head = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD)
        return head + asumOneChain(v, vOff + unrolled, len - unrolled)
    }

    fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        axpyArithmetic(y, yOff, alpha, x, xOff, len)
    }

    fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
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

    /**
     * Exchange two runs a vector at a time. No accumulator, so no dependency chain to break and no unrolled
     * variant: the win is two loads and two stores per vector where the loop does them per element.
     */
    fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val va = DoubleVector.fromArray(SPECIES, a, aOff + i)
            DoubleVector.fromArray(SPECIES, b, bOff + i).intoArray(a, aOff + i)
            va.intoArray(b, bOff + i)
            i += LANE
        }
        while (i < len) {
            val t = a[aOff + i]
            a[aOff + i] = b[bOff + i]
            b[bOff + i] = t
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

    /**
     * Apply the modified Givens matrix ([h11]/[h12]/[h21]/[h22]) to each pair in [x]/[y] a vector at a time,
     * at unit stride only. Both operands of a lane are loaded before either is stored, so this stays correct
     * even when [x] and [y] are the same array at the same offset.
     */
    @Suppress("LongParameterList")
    fun rotm(
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        len: Int,
        h11: Double,
        h12: Double,
        h21: Double,
        h22: Double,
    ) {
        val h11Vec = DoubleVector.broadcast(SPECIES, h11)
        val h12Vec = DoubleVector.broadcast(SPECIES, h12)
        val h21Vec = DoubleVector.broadcast(SPECIES, h21)
        val h22Vec = DoubleVector.broadcast(SPECIES, h22)
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val vx = DoubleVector.fromArray(SPECIES, x, xOff + i)
            val vy = DoubleVector.fromArray(SPECIES, y, yOff + i)
            val newX = vx.fma(h11Vec, vy.mul(h12Vec))
            val newY = vx.fma(h21Vec, vy.mul(h22Vec))
            newX.intoArray(x, xOff + i)
            newY.intoArray(y, yOff + i)
            i += LANE
        }
        while (i < len) {
            val xi = x[xOff + i]
            val yi = y[yOff + i]
            x[xOff + i] = h11 * xi + h12 * yi
            y[yOff + i] = h21 * xi + h22 * yi
            i++
        }
    }
}
