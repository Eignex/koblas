package com.eignex.koblas.dense

import com.eignex.koblas.dense.Simd.UNROLL_MIN
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.numeric.*
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators
import kotlin.math.abs
import kotlin.math.sqrt

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

/**
 * Lane width of the vector path, or 0 when it is unavailable. A run below one lane executes no vector
 * body at all, so short runs take the scalar kernels rather than pay the vector prologue.
 */
private val simdLanes: Int = if (simdAvailable) Simd.lanes() else 0

internal actual object F64PlatformKernels : F64Kernels {
    actual override val name: String
        get() = if (simdAvailable) "${BackendNames.SIMD}($simdLanes lanes)" else BackendNames.C

    override val isPortable: Boolean get() = true

    /**
     * Always true: without the vector module these kernels run the bundled C implementation, so the
     * SIMD probe shows up in [name] instead of here.
     */
    override val isAvailable: Boolean get() = true

    private fun vectorizes(len: Int): Boolean = simdAvailable && len >= simdLanes

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (simdAvailable) {
            if (vectorizes(len)) Simd.dot(a, aOff, b, bOff, len) else scalarDot(a, aOff, b, bOff, len)
        } else {
            JvmCKernelBindings.denseDot(a, aOff, b, bOff, len)
        }

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (vectorizes(len)) {
            Simd.axpy(y, yOff, alpha, x, xOff, len)
        } else if (simdAvailable) {
            scalarAxpy(y, yOff, alpha, x, xOff, len)
        } else {
            JvmCKernelBindings.denseAxpy(y, yOff, alpha, x, xOff, len)
        }
    }

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (vectorizes(len)) {
            Simd.scale(v, vOff, alpha, len)
        } else if (simdAvailable) {
            scalarScale(v, vOff, alpha, len)
        } else {
            JvmCKernelBindings.denseScale(v, vOff, alpha, len)
        }
    }

    /**
     * The plain sum of squares vectorizes; only the rescaling that overflow or underflow forces does not,
     * and that path defers to the shared implementation.
     */
    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double {
        if (!simdAvailable) return JvmCKernelBindings.denseNrm2(v, vOff, len)
        if (vectorizes(len)) {
            val squares = Simd.dot(v, vOff, v, vOff, len)
            if (squares.isFinite() && squares >= F64_MIN_NORMAL) return sqrt(squares)
        }
        return euclideanNorm(v, vOff, len)
    }

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (vectorizes(len)) {
            Simd.swap(a, aOff, b, bOff, len)
        } else if (simdAvailable) {
            super.swap(a, aOff, b, bOff, len)
        } else {
            JvmCKernelBindings.denseSwap(a, aOff, b, bOff, len)
        }
    }

    @Suppress("LongParameterList") // three runs and the multiplier, matching the seam
    override fun symvColumn(
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: Double,
        len: Int,
    ): Double = when {
        vectorizes(len) -> Simd.symvColumn(a, aOff, x, xOff, y, yOff, mult, len)
        simdAvailable -> super.symvColumn(a, aOff, x, xOff, y, yOff, mult, len)
        else -> JvmCKernelBindings.denseSymvColumn(a, aOff, x, xOff, y, yOff, mult, len)
    }

    /** Overridden because [Simd.symvColumn4] reads x and y once for all four columns. */
    @Suppress("LongParameterList") // four column offsets over three runs, matching the seam
    override fun symvColumn4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: DoubleArray,
        out: DoubleArray,
        len: Int,
    ) {
        if (vectorizes(len)) {
            Simd.symvColumn4(a, aOff, stride, x, xOff, y, yOff, mult, out, len)
        } else if (simdAvailable) {
            super.symvColumn4(a, aOff, stride, x, xOff, y, yOff, mult, out, len)
        } else {
            JvmCKernelBindings.denseSymvColumn4(a, aOff, stride, x, xOff, y, yOff, mult, out, len)
        }
    }

    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = if (!simdAvailable) {
        JvmCKernelBindings.denseAsum(v, vOff, len)
    } else if (vectorizes(len)) {
        Simd.asum(v, vOff, len)
    } else {
        absoluteSum(v, vOff, len)
    }

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
        } else if (simdAvailable) {
            scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
        } else {
            JvmCKernelBindings.denseDot4(a, aOff, stride, b, bOff, len, out, outOff)
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

    /** One pass over the column, accumulating the dot while writing y, so `a` is loaded once. */
    @Suppress("LongParameterList") // three runs and the multiplier, matching the seam
    fun symvColumn(
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: Double,
        len: Int,
    ): Double {
        var acc = DoubleVector.zero(SPECIES)
        val vm = DoubleVector.broadcast(SPECIES, mult)
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val va = DoubleVector.fromArray(SPECIES, a, aOff + i)
            acc = va.fma(DoubleVector.fromArray(SPECIES, x, xOff + i), acc)
            va.fma(vm, DoubleVector.fromArray(SPECIES, y, yOff + i)).intoArray(y, yOff + i)
            i += LANE
        }
        var s = acc.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            val ai = a[aOff + i]
            s += ai * x[xOff + i]
            y[yOff + i] += mult * ai
            i++
        }
        return s
    }

    /** One pass over four columns, so x is loaded once and y is loaded and stored once for the four. */
    @Suppress("LongParameterList") // four column offsets over three runs, matching the seam
    fun symvColumn4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: DoubleArray,
        out: DoubleArray,
        len: Int,
    ) {
        val o1 = aOff + stride
        val o2 = aOff + 2 * stride
        val o3 = aOff + 3 * stride
        val f0 = mult[0]
        val f1 = mult[1]
        val f2 = mult[2]
        val f3 = mult[3]
        val m0 = DoubleVector.broadcast(SPECIES, f0)
        val m1 = DoubleVector.broadcast(SPECIES, f1)
        val m2 = DoubleVector.broadcast(SPECIES, f2)
        val m3 = DoubleVector.broadcast(SPECIES, f3)
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val vx = DoubleVector.fromArray(SPECIES, x, xOff + i)
            val a0 = DoubleVector.fromArray(SPECIES, a, aOff + i)
            val a1 = DoubleVector.fromArray(SPECIES, a, o1 + i)
            val a2 = DoubleVector.fromArray(SPECIES, a, o2 + i)
            val a3 = DoubleVector.fromArray(SPECIES, a, o3 + i)
            s0 = a0.fma(vx, s0)
            s1 = a1.fma(vx, s1)
            s2 = a2.fma(vx, s2)
            s3 = a3.fma(vx, s3)
            var vy = DoubleVector.fromArray(SPECIES, y, yOff + i)
            vy = a0.fma(m0, vy)
            vy = a1.fma(m1, vy)
            vy = a2.fma(m2, vy)
            vy = a3.fma(m3, vy)
            vy.intoArray(y, yOff + i)
            i += LANE
        }
        var r0 = s0.reduceLanes(VectorOperators.ADD)
        var r1 = s1.reduceLanes(VectorOperators.ADD)
        var r2 = s2.reduceLanes(VectorOperators.ADD)
        var r3 = s3.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            val xi = x[xOff + i]
            val q0 = a[aOff + i]
            val q1 = a[o1 + i]
            val q2 = a[o2 + i]
            val q3 = a[o3 + i]
            r0 += q0 * xi
            r1 += q1 * xi
            r2 += q2 * xi
            r3 += q3 * xi
            y[yOff + i] += q0 * f0 + q1 * f1 + q2 * f2 + q3 * f3
            i++
        }
        out[0] = r0
        out[1] = r1
        out[2] = r2
        out[3] = r3
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
}
