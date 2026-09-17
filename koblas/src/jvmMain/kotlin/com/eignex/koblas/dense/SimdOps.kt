package com.eignex.koblas.dense

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
    // Four accumulators pay for the reduce that combines them from this many vectors on.
    private val UNROLL_MIN = 32 * LANE

    /**
     * One accumulator chains every addition on the previous one's result, and an add's latency is several
     * times its throughput, so a single chain leaves most of the unit idle on a long run. Four independent
     * chains keep it fed, which is what the long-run arm below runs.
     *
     * The product and the sum are separate operations rather than one fused multiply-add. BLAS requires the
     * multiplication to round before the addition, which a fused one does not do, so the two are different
     * results and only this one is the documented answer. A fused multiply-add is also not an instruction
     * everywhere: a machine without one gets the Vector API's software fallback, which is a `Math.fma` call
     * per lane and runs a thousand times slower than the scalar loop it was meant to beat.
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
            sum = va.mul(vb).add(sum)
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
                .mul(DoubleVector.fromArray(SPECIES, b, bOff + i)).add(s0)
            s1 = DoubleVector.fromArray(SPECIES, a, aOff + i + LANE)
                .mul(DoubleVector.fromArray(SPECIES, b, bOff + i + LANE)).add(s1)
            s2 = DoubleVector.fromArray(SPECIES, a, aOff + i + 2 * LANE)
                .mul(DoubleVector.fromArray(SPECIES, b, bOff + i + 2 * LANE)).add(s2)
            s3 = DoubleVector.fromArray(SPECIES, a, aOff + i + 3 * LANE)
                .mul(DoubleVector.fromArray(SPECIES, b, bOff + i + 3 * LANE)).add(s3)
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
     * A lane-wise absolute value, as a mask over the sign bit rather than a branch.
     *
     * Inline because a helper returning a [DoubleVector] the JIT declines to inline makes the vector a heap
     * object, and the loop then allocates once per iteration. Kotlin's warning does not account for avoiding
     * that allocation.
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
}
