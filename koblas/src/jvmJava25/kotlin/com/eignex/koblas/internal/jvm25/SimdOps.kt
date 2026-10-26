package com.eignex.koblas.internal.jvm25

import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.LongVector
import jdk.incubator.vector.VectorOperators
import kotlin.math.abs

internal object SimdOps {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED

    /** Every bit but the sign bit of a double. */
    private const val SIGN_MASK = 0x7fffffffffffffffL
    private val LANE = SPECIES.length()
    private val LONG_SPECIES = SPECIES.withLanes(Long::class.java)
    private val SIGN_MASK_BITS = LongArray(LANE) { SIGN_MASK }

    fun lanes(): Int = LANE

    /** Run length for four accumulators, scaled by the machine's lane count. */
    private val UNROLL_MIN = 32 * LANE

    /**
     * Four independent accumulators hide addition latency on long runs. [multiplyAdd] uses FMA only
     * when [hardwareFusedMultiplyAdd] enables the instruction.
     *
     * Keep short and unrolled bodies separate so the short body remains small enough for JIT inlining.
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
            sum = multiplyAdd(va, vb, sum)
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
            s0 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aOff + i),
                DoubleVector.fromArray(SPECIES, b, bOff + i),
                s0,
            )
            s1 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aOff + i + LANE),
                DoubleVector.fromArray(SPECIES, b, bOff + i + LANE),
                s1,
            )
            s2 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aOff + i + 2 * LANE),
                DoubleVector.fromArray(SPECIES, b, bOff + i + 2 * LANE),
                s2,
            )
            s3 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aOff + i + 3 * LANE),
                DoubleVector.fromArray(SPECIES, b, bOff + i + 3 * LANE),
                s3,
            )
            i += 4 * LANE
        }
        val head = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD)
        return head + dotOneChain(a, aOff + unrolled, b, bOff + unrolled, len - unrolled)
    }

    /**
     * Skip blocks whose magnitudes cannot improve the current winner. Only a block with a greater
     * magnitude needs an ordered scalar search. Repeated improvements switch to the unconditional
     * reduction, so an increasing input does not make every block run both scans.
     */
    fun iamax(v: DoubleArray, vOff: Int, len: Int): Int =
        if (len >= UNROLL_MIN) iamaxFiltered(v, vOff, len) else iamaxReduced(v, vOff, len)

    private fun iamaxFiltered(v: DoubleArray, vOff: Int, len: Int): Int {
        val mask = LongVector.fromArray(LONG_SPECIES, SIGN_MASK_BITS, 0)
        var threshold = DoubleVector.zero(SPECIES)
        var bestAbs = 0.0
        var winner = 0
        var improvingBlocks = 0
        var index = vOff
        val end = vOff + len
        val blockEnd = end - len % (4 * LANE)
        while (index < blockEnd) {
            val a = signStripped(DoubleVector.fromArray(SPECIES, v, index), mask)
            val b = signStripped(DoubleVector.fromArray(SPECIES, v, index + LANE), mask)
            val c = signStripped(DoubleVector.fromArray(SPECIES, v, index + 2 * LANE), mask)
            val d = signStripped(DoubleVector.fromArray(SPECIES, v, index + 3 * LANE), mask)
            val candidates = a.compare(VectorOperators.GT, threshold).or(b.compare(VectorOperators.GT, threshold))
                .or(c.compare(VectorOperators.GT, threshold).or(d.compare(VectorOperators.GT, threshold)))
            if (candidates.anyTrue()) {
                val next = index + 4 * LANE
                while (index < next) {
                    val magnitude = abs(v[index])
                    if (magnitude > bestAbs) {
                        bestAbs = magnitude
                        winner = index - vOff
                    }
                    index++
                }
                improvingBlocks++
                if (improvingBlocks == 4 && index < end) {
                    val remaining = index + iamaxReduced(v, index, end - index)
                    return if (abs(v[remaining]) > bestAbs) remaining - vOff else winner
                }
                threshold = DoubleVector.broadcast(SPECIES, bestAbs)
            } else {
                improvingBlocks = 0
                index += 4 * LANE
            }
        }
        while (index < end) {
            val magnitude = abs(v[index])
            if (magnitude > bestAbs) {
                bestAbs = magnitude
                winner = index - vOff
            }
            index++
        }
        return winner
    }

    /** Unconditional magnitude reduction and ordered index search for short or frequently improving input. */
    private fun iamaxReduced(v: DoubleArray, vOff: Int, len: Int): Int {
        if (len == 0) return -1
        val mask = LongVector.fromArray(LONG_SPECIES, SIGN_MASK_BITS, 0)
        val bound = vOff + SPECIES.loopBound(len)
        val end = vOff + len
        var maximum = DoubleVector.zero(SPECIES)
        var second = DoubleVector.zero(SPECIES)
        var third = DoubleVector.zero(SPECIES)
        var fourth = DoubleVector.zero(SPECIES)
        var index = vOff
        val quadEnd = vOff + len - len % (4 * LANE)
        while (index < quadEnd) {
            val a = signStripped(DoubleVector.fromArray(SPECIES, v, index), mask)
            val b = signStripped(DoubleVector.fromArray(SPECIES, v, index + LANE), mask)
            val c = signStripped(DoubleVector.fromArray(SPECIES, v, index + 2 * LANE), mask)
            val d = signStripped(DoubleVector.fromArray(SPECIES, v, index + 3 * LANE), mask)
            maximum = maximum.blend(a, a.compare(VectorOperators.GT, maximum))
            second = second.blend(b, b.compare(VectorOperators.GT, second))
            third = third.blend(c, c.compare(VectorOperators.GT, third))
            fourth = fourth.blend(d, d.compare(VectorOperators.GT, fourth))
            index += 4 * LANE
        }
        maximum = maximum.max(second).max(third.max(fourth))
        while (index < bound) {
            val values = signStripped(DoubleVector.fromArray(SPECIES, v, index), mask)
            maximum = maximum.blend(values, values.compare(VectorOperators.GT, maximum))
            index += LANE
        }
        var bestAbs = maximum.reduceLanes(VectorOperators.MAX)
        while (index < end) {
            val magnitude = abs(v[index])
            if (magnitude > bestAbs) bestAbs = magnitude
            index++
        }
        if (bestAbs == 0.0) return 0
        index = vOff
        while (index < bound) {
            val matches = signStripped(DoubleVector.fromArray(SPECIES, v, index), mask).eq(bestAbs)
            if (matches.anyTrue()) return index - vOff + matches.firstTrue()
            index += LANE
        }
        while (index < end) {
            if (abs(v[index]) == bestAbs) return index - vOff
            index++
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
     * One multiply-add, fused where [hardwareFusedMultiplyAdd] found the instruction and two operations
     * where it did not.
     *
     * Inline for the reason [signStripped] gives, and because the test is then a constant the JIT folds, so
     * one of the two forms reaches the loop and neither a branch nor the other form survives in it.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun multiplyAdd(x: DoubleVector, y: DoubleVector, accumulator: DoubleVector): DoubleVector =
        if (hardwareFusedMultiplyAdd) x.fma(y, accumulator) else x.mul(y).add(accumulator)

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

    /** Inline for the same carrier-allocation reason as the scalar-mask overload. */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun signStripped(v: DoubleVector, mask: LongVector): DoubleVector =
        v.reinterpretAsLongs().and(mask).reinterpretAsDoubles()

    /**
     * Explicit lanes permit the reordered floating-point reduction that HotSpot will not generate
     * from a scalar loop. Four accumulators hide dependency latency as in [dot].
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
        val head = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD)
        return head + asumOneChain(v, vOff + unrolled, len - unrolled)
    }
}
