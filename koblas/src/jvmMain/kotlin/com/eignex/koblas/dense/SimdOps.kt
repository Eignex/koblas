package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.scalarDot
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators
import kotlin.math.abs

internal object SimdOps {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED

    /** Every bit but the sign bit of a double. */
    private const val SIGN_MASK = 0x7fffffffffffffffL
    private val LANE = SPECIES.length()

    fun lanes(): Int = LANE

    /** Run length for four accumulators, scaled by the machine's lane count. */
    private val UNROLL_MIN = DenseTuning.simdUnrollMinVectors * LANE

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
     * Reduce magnitudes independently in each lane, then locate the first match in input order.
     * Separating the index search removes the scalar value/index dependency from the reduction. Four
     * independent vectors break the remaining maximum dependency chain. Strict comparisons leave NaNs
     * out without changing ties or the all-zero result.
     */
    fun iamax(v: DoubleArray, vOff: Int, len: Int): Int {
        if (len == 0) return -1
        val bound = SPECIES.loopBound(len)
        var maximum = DoubleVector.zero(SPECIES)
        var second = DoubleVector.zero(SPECIES)
        var third = DoubleVector.zero(SPECIES)
        var fourth = DoubleVector.zero(SPECIES)
        var i = 0
        val quadBound = len - len % (4 * LANE)
        while (i < quadBound) {
            val a = DoubleVector.fromArray(SPECIES, v, vOff + i).abs()
            val b = DoubleVector.fromArray(SPECIES, v, vOff + i + LANE).abs()
            val c = DoubleVector.fromArray(SPECIES, v, vOff + i + 2 * LANE).abs()
            val d = DoubleVector.fromArray(SPECIES, v, vOff + i + 3 * LANE).abs()
            maximum = maximum.blend(a, a.compare(VectorOperators.GT, maximum))
            second = second.blend(b, b.compare(VectorOperators.GT, second))
            third = third.blend(c, c.compare(VectorOperators.GT, third))
            fourth = fourth.blend(d, d.compare(VectorOperators.GT, fourth))
            i += 4 * LANE
        }
        maximum = maximum.max(second).max(third.max(fourth))
        while (i < bound) {
            val values = DoubleVector.fromArray(SPECIES, v, vOff + i).abs()
            maximum = maximum.blend(values, values.compare(VectorOperators.GT, maximum))
            i += LANE
        }
        var bestAbs = maximum.reduceLanes(VectorOperators.MAX)
        while (i < len) {
            val magnitude = abs(v[vOff + i])
            if (magnitude > bestAbs) bestAbs = magnitude
            i++
        }
        if (bestAbs == 0.0) return 0
        i = 0
        while (i < bound) {
            val matches = DoubleVector.fromArray(SPECIES, v, vOff + i).abs().eq(bestAbs)
            if (matches.anyTrue()) return i + matches.firstTrue()
            i += LANE
        }
        while (i < len) {
            if (abs(v[vOff + i]) == bestAbs) return i
            i++
        }
        return 0
    }

    /** Four chains above [UNROLL_MIN], one accumulator below it, and a scalar tail. */
    fun sum(v: DoubleArray, vOff: Int, len: Int): Double {
        var i = 0
        var s = 0.0
        if (len >= UNROLL_MIN) {
            val quadBound = len - (len % (4 * LANE))
            var a = DoubleVector.zero(SPECIES)
            var b = DoubleVector.zero(SPECIES)
            var c = DoubleVector.zero(SPECIES)
            var d = DoubleVector.zero(SPECIES)
            while (i < quadBound) {
                a = a.add(DoubleVector.fromArray(SPECIES, v, vOff + i))
                b = b.add(DoubleVector.fromArray(SPECIES, v, vOff + i + LANE))
                c = c.add(DoubleVector.fromArray(SPECIES, v, vOff + i + 2 * LANE))
                d = d.add(DoubleVector.fromArray(SPECIES, v, vOff + i + 3 * LANE))
                i += 4 * LANE
            }
            s = a.add(b).add(c.add(d)).reduceLanes(VectorOperators.ADD)
        }
        val bound = SPECIES.loopBound(len)
        var total = DoubleVector.zero(SPECIES)
        while (i < bound) {
            total = total.add(DoubleVector.fromArray(SPECIES, v, vOff + i))
            i += LANE
        }
        s += total.reduceLanes(VectorOperators.ADD)
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
     * accumulators. [DenseBlas.gemv] wants this over four [dot] calls and their four reductions.
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

    @Suppress("LongParameterList")
    fun axpy4(
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
        val vc0 = DoubleVector.broadcast(SPECIES, c0)
        val vc1 = DoubleVector.broadcast(SPECIES, c1)
        val vc2 = DoubleVector.broadcast(SPECIES, c2)
        val vc3 = DoubleVector.broadcast(SPECIES, c3)
        val o1 = aOff + stride
        val o2 = aOff + 2 * stride
        val o3 = aOff + 3 * stride
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            var value = DoubleVector.fromArray(SPECIES, y, yOff + i)
            value = DoubleVector.fromArray(SPECIES, a, aOff + i).fma(vc0, value)
            value = DoubleVector.fromArray(SPECIES, a, o1 + i).fma(vc1, value)
            value = DoubleVector.fromArray(SPECIES, a, o2 + i).fma(vc2, value)
            value = DoubleVector.fromArray(SPECIES, a, o3 + i).fma(vc3, value)
            value.intoArray(y, yOff + i)
            i += LANE
        }
        while (i < len) {
            var value = y[yOff + i]
            value += c0 * a[aOff + i]
            value += c1 * a[o1 + i]
            value += c2 * a[o2 + i]
            value += c3 * a[o3 + i]
            y[yOff + i] = value
            i++
        }
    }

    @Suppress("LongParameterList")
    fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double {
        val alphaVector = DoubleVector.broadcast(SPECIES, alpha)
        // If every product is smaller than this bound, even their absolute sum stays finite, so splitting
        // the reduction across lanes cannot hide or introduce intermediate overflow. The rare exceptional
        // path below replays the dot in scalar encounter order after retaining the SIMD AXPY update.
        val safeProductLimit = if (len == 0) Double.MAX_VALUE else Double.MAX_VALUE / len
        var requiresOrderedDot = false
        var sum = DoubleVector.zero(SPECIES)
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val va = DoubleVector.fromArray(SPECIES, a, aOff + i)
            val vx = DoubleVector.fromArray(SPECIES, x, xOff + i)
            val products = va.mul(vx)
            requiresOrderedDot = requiresOrderedDot ||
                signStripped(products).compare(VectorOperators.GE, safeProductLimit).anyTrue()
            sum = products.add(sum)
            va.mul(alphaVector).add(DoubleVector.fromArray(SPECIES, y, yOff + i)).intoArray(y, yOff + i)
            i += LANE
        }
        var result = sum.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            val ai = a[aOff + i]
            val xi = x[xOff + i]
            requiresOrderedDot = requiresOrderedDot || abs(ai * xi) >= safeProductLimit
            result += ai * xi
            y[yOff + i] += alpha * ai
            i++
        }
        return if (requiresOrderedDot) scalarDot(a, aOff, x, xOff, len) else result
    }

    /**
     * A vector with every sign bit cleared, which is the absolute value of each lane.
     *
     * The result is identical for every input, including negative zero, infinities and NaN.
     *
     * The `inline` keyword is load-bearing and must stay. A function returning a vector is one the JIT has
     * to inline for the value to live in a register; otherwise the loop allocates a heap vector.
     *
     * Kotlin's warning does not account for avoiding that allocation.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun signStripped(v: DoubleVector): DoubleVector =
        v.reinterpretAsLongs().lanewise(VectorOperators.AND, SIGN_MASK).reinterpretAsDoubles()

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
            sum = sum.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i)))
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
            s0 = s0.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i)))
            s1 = s1.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i + LANE)))
            s2 = s2.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i + 2 * LANE)))
            s3 = s3.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i + 3 * LANE)))
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
            // Parent BLAS routines require the multiplication to round before the addition. A fused
            // operation can hide an overflowing product and change a later cancellation.
            vx.mul(alphaVec).add(vy).intoArray(y, yOff + i)
            i += LANE
        }
        while (i < len) {
            y[yOff + i] += alpha * x[xOff + i]
            i++
        }
    }

    /** Exchange two runs a vector at a time. */
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
        // Four independent loads and stores amortize the Vector API loop checks.
        val unrolledBound = len - 4 * LANE + 1
        while (i < unrolledBound) {
            val v0 = DoubleVector.fromArray(SPECIES, v, vOff + i)
            val v1 = DoubleVector.fromArray(SPECIES, v, vOff + i + LANE)
            val v2 = DoubleVector.fromArray(SPECIES, v, vOff + i + 2 * LANE)
            val v3 = DoubleVector.fromArray(SPECIES, v, vOff + i + 3 * LANE)
            v0.mul(alphaVec).intoArray(v, vOff + i)
            v1.mul(alphaVec).intoArray(v, vOff + i + LANE)
            v2.mul(alphaVec).intoArray(v, vOff + i + 2 * LANE)
            v3.mul(alphaVec).intoArray(v, vOff + i + 3 * LANE)
            i += 4 * LANE
        }
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
