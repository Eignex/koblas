package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.IntVector
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
    private val INT_SPECIES = SPECIES.withLanes(Int::class.java)
    private val HIGH_WORD_MASK_BITS = LongArray(LANE) { 0x7fffffff00000000L }

    fun lanes(): Int = LANE

    /** Contiguous AXPY with compact and streaming bodies. */
    @Suppress("LongParameterList") // two independently addressed vector windows and their multiplier
    fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        if (len >= WIDE_AXPY_MIN) {
            axpyUnrolled(y, yOff, alpha, x, xOff, len)
        } else if (LANE >= 8 && len >= 2 * LANE && len % LANE >= LANE / 2) {
            axpyTail(y, yOff, alpha, x, xOff, len)
        } else {
            axpyShort(y, yOff, alpha, x, xOff, len)
        }
    }

    /** A scalar remainder is cheaper than another vector block for a tail shorter than half a vector. */
    @Suppress("LongParameterList") // two independently addressed vector windows and their multiplier
    private fun axpyShort(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        val multiplier = DoubleVector.broadcast(SPECIES, alpha)
        val bound = SPECIES.loopBound(len)
        var i = 0
        while (i < bound) {
            DoubleVector.fromArray(SPECIES, x, xOff + i).mul(multiplier)
                .add(DoubleVector.fromArray(SPECIES, y, yOff + i)).intoArray(y, yOff + i)
            i += LANE
        }
        while (i < len) {
            y[yOff + i] += alpha * x[xOff + i]
            i++
        }
    }

    /** Multiple full vectors amortize a final block that also covers a substantial tail. */
    @Suppress("LongParameterList") // two independently addressed vector windows and their multiplier
    private fun axpyTail(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        val multiplier = DoubleVector.broadcast(SPECIES, alpha)
        val last = len - LANE
        // The last block can overlap the loop. Compute it before any stores so that overlap is updated once.
        val finalValues = DoubleVector.fromArray(SPECIES, x, xOff + last).mul(multiplier)
            .add(DoubleVector.fromArray(SPECIES, y, yOff + last))
        var i = 0
        while (i < last) {
            DoubleVector.fromArray(SPECIES, x, xOff + i).mul(multiplier)
                .add(DoubleVector.fromArray(SPECIES, y, yOff + i)).intoArray(y, yOff + i)
            i += LANE
        }
        finalValues.intoArray(y, yOff + last)
    }

    /** Long streaming calls amortize four independent vector updates per iteration. */
    @Suppress("LongParameterList") // two independently addressed vector windows and their multiplier
    private fun axpyUnrolled(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        val multiplier = DoubleVector.broadcast(SPECIES, alpha)
        val bound = SPECIES.loopBound(len)
        val unrolled = len - len % (4 * LANE)
        var i = 0
        while (i < unrolled) {
            DoubleVector.fromArray(SPECIES, x, xOff + i).mul(multiplier)
                .add(DoubleVector.fromArray(SPECIES, y, yOff + i)).intoArray(y, yOff + i)
            DoubleVector.fromArray(SPECIES, x, xOff + i + LANE).mul(multiplier)
                .add(DoubleVector.fromArray(SPECIES, y, yOff + i + LANE)).intoArray(y, yOff + i + LANE)
            DoubleVector.fromArray(SPECIES, x, xOff + i + 2 * LANE).mul(multiplier)
                .add(DoubleVector.fromArray(SPECIES, y, yOff + i + 2 * LANE)).intoArray(y, yOff + i + 2 * LANE)
            DoubleVector.fromArray(SPECIES, x, xOff + i + 3 * LANE).mul(multiplier)
                .add(DoubleVector.fromArray(SPECIES, y, yOff + i + 3 * LANE)).intoArray(y, yOff + i + 3 * LANE)
            i += 4 * LANE
        }
        while (i < bound) {
            val increment = DoubleVector.fromArray(SPECIES, x, xOff + i).mul(multiplier)
            increment.add(DoubleVector.fromArray(SPECIES, y, yOff + i)).intoArray(y, yOff + i)
            i += LANE
        }
        while (i < len) {
            y[yOff + i] += alpha * x[xOff + i]
            i++
        }
    }

    /** Run length for four accumulators, scaled by the machine's lane count. */
    private val UNROLL_MIN = 32 * LANE

    // Peeling trades aligned loads for scalar work at both ends; short windows do not amortise it.
    private const val ALIGNMENT_MIN = 4_096
    private const val IAMAX_ALIGNMENT_MIN = 65_536
    private const val WIDE_ASUM_MIN = 65_536
    private const val WIDE_AXPY_MIN = 65_536

    /**
     * Four independent accumulators hide addition latency on long runs. [multiplyAdd] uses FMA only
     * when [hardwareFusedMultiplyAdd] enables the instruction.
     *
     * Keep short and unrolled bodies separate so the short body remains small enough for JIT inlining.
     */
    fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = when {
        len >= ALIGNMENT_MIN -> dotAligned(a, aOff, b, bOff, len)
        len >= UNROLL_MIN -> dotUnrolled(a, aOff, b, bOff, len)
        else -> dotOneChain(a, aOff, b, bOff, len)
    }

    private fun dotAligned(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        val prefix = HotSpotArrayAlignment.prefix(a, aOff, len)
        if (prefix == 0) return dotUnrolled(a, aOff, b, bOff, len)
        var head = 0.0
        for (i in 0 until prefix) head += a[aOff + i] * b[bOff + i]
        return head + dotUnrolled(a, aOff + prefix, b, bOff + prefix, len - prefix)
    }

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

    /**
     * [dot] past [UNROLL_MIN], where the four chains pay for the reduce that combines them.
     * Absolute indices let C2 fold the window offsets rather than reload them inside the loop.
     * Each iteration updates every chain twice to amortise address and loop bookkeeping.
     */
    private fun dotUnrolled(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        if (NARROW) return dotEightChains(a, aOff, b, bOff, len)
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        val unrolled = len - len % (8 * LANE)
        val aEnd = aOff + unrolled
        var aIndex = aOff
        var bIndex = bOff
        while (aIndex < aEnd) {
            s0 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex),
                DoubleVector.fromArray(SPECIES, b, bIndex),
                s0,
            )
            s1 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + LANE),
                s1,
            )
            s2 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 2 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 2 * LANE),
                s2,
            )
            s3 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 3 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 3 * LANE),
                s3,
            )
            s0 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 4 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 4 * LANE),
                s0,
            )
            s1 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 5 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 5 * LANE),
                s1,
            )
            s2 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 6 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 6 * LANE),
                s2,
            )
            s3 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 7 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 7 * LANE),
                s3,
            )
            aIndex += 8 * LANE
            bIndex += 8 * LANE
        }
        val head = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD)
        // A bounded scalar remainder avoids another short Vector API reduction's inlining budget.
        var tail = 0.0
        val end = aOff + len
        while (aIndex < end) {
            tail += a[aIndex] * b[bIndex]
            aIndex++
            bIndex++
        }
        return head + tail
    }

    /**
     * Skip blocks whose magnitudes cannot improve the current winner. Only a block with a greater
     * magnitude needs an ordered scalar search. Repeated improvements switch to the unconditional
     * reduction, so an increasing input does not make every block run both scans.
     */
    fun iamax(v: DoubleArray, vOff: Int, len: Int): Int = when {
        len >= IAMAX_ALIGNMENT_MIN -> iamaxAligned(v, vOff, len)
        len >= UNROLL_MIN -> iamaxFiltered(v, vOff, len)
        else -> iamaxReduced(v, vOff, len)
    }

    private fun iamaxAligned(v: DoubleArray, vOff: Int, len: Int): Int {
        val prefix = HotSpotArrayAlignment.prefix(v, vOff, len)
        if (prefix == 0) return iamaxFiltered(v, vOff, len)
        var bestAbs = 0.0
        var winner = 0
        for (i in 0 until prefix) {
            val magnitude = abs(v[vOff + i])
            if (magnitude > bestAbs) {
                bestAbs = magnitude
                winner = i
            }
        }
        val remaining = prefix + iamaxFiltered(v, vOff + prefix, len - prefix)
        return if (abs(v[vOff + remaining]) > bestAbs) remaining else winner
    }

    private fun iamaxFiltered(v: DoubleArray, vOff: Int, len: Int): Int {
        if (LANE == 2) return iamaxNarrowFiltered(v, vOff, len)
        val mask = LongVector.fromArray(LONG_SPECIES, SIGN_MASK_BITS, 0)
        var threshold = DoubleVector.zero(SPECIES)
        var bestAbs = 0.0
        var winner = 0
        var improvingBlocks = 0
        var index = vOff
        val end = vOff + len
        // Eight vectors fold into one maximum and a single compare; a compare per vector spent mask operations
        // on the ports the loads and the sign strip also need.
        val blockEnd = end - len % (8 * LANE)
        while (index < blockEnd) {
            val a = signStripped(DoubleVector.fromArray(SPECIES, v, index), mask)
            val b = signStripped(DoubleVector.fromArray(SPECIES, v, index + LANE), mask)
            val c = signStripped(DoubleVector.fromArray(SPECIES, v, index + 2 * LANE), mask)
            val d = signStripped(DoubleVector.fromArray(SPECIES, v, index + 3 * LANE), mask)
            val e = signStripped(DoubleVector.fromArray(SPECIES, v, index + 4 * LANE), mask)
            val f = signStripped(DoubleVector.fromArray(SPECIES, v, index + 5 * LANE), mask)
            val g = signStripped(DoubleVector.fromArray(SPECIES, v, index + 6 * LANE), mask)
            val h = signStripped(DoubleVector.fromArray(SPECIES, v, index + 7 * LANE), mask)
            val maximum = a.max(b).max(c.max(d)).max(e.max(f).max(g.max(h)))
            // A NaN lane fails the comparison, so its block takes the ordered scan, which skips it as before.
            if (!maximum.compare(VectorOperators.LE, threshold).allTrue()) {
                val next = index + 8 * LANE
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
                index += 8 * LANE
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

    /**
     * HotSpot on x86 cannot intrinsify two-lane masks. Four integer lanes hold the high words of two
     * magnitudes, with the low words cleared. A threshold one below the winner's high word admits
     * ties in that word so the scalar search still distinguishes all 52 fraction bits. Repeated
     * candidates fall back to scalar work to bound rescans without reaching another two-lane mask.
     */
    private fun iamaxNarrowFiltered(v: DoubleArray, vOff: Int, len: Int): Int {
        val maskBits = LongVector.fromArray(LONG_SPECIES, HIGH_WORD_MASK_BITS, 0)
        val mask = maskBits.reinterpretShape(INT_SPECIES, 0) as IntVector
        var threshold = IntVector.broadcast(INT_SPECIES, -1)
        var bestAbs = 0.0
        var winner = 0
        var candidateBlocks = 0
        var index = vOff
        val end = vOff + len
        val blockEnd = end - len % (8 * LANE)
        while (index < blockEnd) {
            val a = narrowSignStripped(DoubleVector.fromArray(SPECIES, v, index), mask)
            val b = narrowSignStripped(DoubleVector.fromArray(SPECIES, v, index + LANE), mask)
            val c = narrowSignStripped(DoubleVector.fromArray(SPECIES, v, index + 2 * LANE), mask)
            val d = narrowSignStripped(DoubleVector.fromArray(SPECIES, v, index + 3 * LANE), mask)
            val e = narrowSignStripped(DoubleVector.fromArray(SPECIES, v, index + 4 * LANE), mask)
            val f = narrowSignStripped(DoubleVector.fromArray(SPECIES, v, index + 5 * LANE), mask)
            val g = narrowSignStripped(DoubleVector.fromArray(SPECIES, v, index + 6 * LANE), mask)
            val h = narrowSignStripped(DoubleVector.fromArray(SPECIES, v, index + 7 * LANE), mask)
            val maximum = a.max(b).max(c.max(d)).max(e.max(f).max(g.max(h)))
            if (maximum.compare(VectorOperators.GT, threshold).anyTrue()) {
                val next = index + 8 * LANE
                while (index < next) {
                    val magnitude = abs(v[index])
                    if (magnitude > bestAbs) {
                        bestAbs = magnitude
                        winner = index - vOff
                    }
                    index++
                }
                candidateBlocks++
                if (candidateBlocks == 4 && index < end) {
                    val remaining = index + ScalarVectorKernels.iamax(v, index, end - index, 1)
                    return if (abs(v[remaining]) > bestAbs) remaining - vOff else winner
                }
                threshold = IntVector.broadcast(INT_SPECIES, (bestAbs.toRawBits() ushr 32).toInt() - 1)
            } else {
                candidateBlocks = 0
                index += 8 * LANE
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
        if (LANE == 2) return ScalarVectorKernels.iamax(v, vOff, len, 1)
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
     * A lane-wise absolute value, as a mask over the sign bit rather than a branch.
     *
     * Inline because a helper returning a [DoubleVector] the JIT declines to inline makes the vector a heap
     * object, and the loop then allocates once per iteration. Kotlin's warning does not account for avoiding
     * that allocation. Explicit species also let C2 eliminate the reinterpretation carrier arrays.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun signStripped(v: DoubleVector): DoubleVector = (v.reinterpretShape(LONG_SPECIES, 0) as LongVector)
        .lanewise(VectorOperators.AND, SIGN_MASK).reinterpretShape(SPECIES, 0) as DoubleVector

    /** Inline for the same carrier-allocation reason as the scalar-mask overload. */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun signStripped(v: DoubleVector, mask: LongVector): DoubleVector =
        (v.reinterpretShape(LONG_SPECIES, 0) as LongVector).and(mask).reinterpretShape(SPECIES, 0) as DoubleVector

    /** Inline so the returned integer vector remains a register value at the narrow filter's load site. */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun narrowSignStripped(v: DoubleVector, mask: IntVector): IntVector =
        (v.reinterpretShape(INT_SPECIES, 0) as IntVector).and(mask)

    /**
     * Explicit lanes permit the reordered floating-point reduction that HotSpot will not generate
     * from a scalar loop. Four accumulators hide dependency latency as in [dot].
     */
    fun asum(v: DoubleArray, vOff: Int, len: Int): Double = when {
        len >= ALIGNMENT_MIN -> asumAligned(v, vOff, len)
        len >= UNROLL_MIN -> asumUnrolled(v, vOff, len)
        else -> asumOneChain(v, vOff, len)
    }

    private fun asumAligned(v: DoubleArray, vOff: Int, len: Int): Double {
        val wide = LANE >= 8 && len >= WIDE_ASUM_MIN && HotSpotArrayAlignment.hasKnownAlignment(v)
        val prefix = HotSpotArrayAlignment.prefix(v, vOff, len)
        if (prefix == 0) return if (wide) asumWideUnrolled(v, vOff, len) else asumUnrolled(v, vOff, len)
        var head = 0.0
        for (i in 0 until prefix) head += abs(v[vOff + i])
        return head + if (wide) {
            asumWideUnrolled(v, vOff + prefix, len - prefix)
        } else {
            asumUnrolled(v, vOff + prefix, len - prefix)
        }
    }

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
        if (NARROW) return asumEightChains(v, vOff, len)
        val mask = LongVector.fromArray(LONG_SPECIES, SIGN_MASK_BITS, 0)
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        var i = 0
        val unrolled = len - len % (4 * LANE)
        while (i < unrolled) {
            s0 = s0.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i), mask))
            s1 = s1.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i + LANE), mask))
            s2 = s2.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i + 2 * LANE), mask))
            s3 = s3.add(signStripped(DoubleVector.fromArray(SPECIES, v, vOff + i + 3 * LANE), mask))
            i += 4 * LANE
        }
        val head = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD)
        // Keep the bounded remainder free of Vector API carrier allocations even before it is compiled.
        var tail = 0.0
        while (i < len) {
            tail += abs(v[vOff + i])
            i++
        }
        return head + tail
    }

    /** Eight loads amortise loop bookkeeping on long runs of wide vectors without adding accumulator chains. */
    private fun asumWideUnrolled(v: DoubleArray, vOff: Int, len: Int): Double {
        val mask = LongVector.fromArray(LONG_SPECIES, SIGN_MASK_BITS, 0)
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        val bound = vOff + len - len % (8 * LANE)
        var index = vOff
        while (index < bound) {
            s0 = s0.add(signStripped(DoubleVector.fromArray(SPECIES, v, index), mask))
            s1 = s1.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + LANE), mask))
            s2 = s2.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 2 * LANE), mask))
            s3 = s3.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 3 * LANE), mask))
            s0 = s0.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 4 * LANE), mask))
            s1 = s1.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 5 * LANE), mask))
            s2 = s2.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 6 * LANE), mask))
            s3 = s3.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 7 * LANE), mask))
            index += 8 * LANE
        }
        val head = s0.add(s1).add(s2.add(s3)).reduceLanes(VectorOperators.ADD)
        var tail = 0.0
        val end = vOff + len
        while (index < end) {
            tail += abs(v[index])
            index++
        }
        return head + tail
    }

    /**
     * Whether a long reduction carries eight chains rather than four. A vector of two lanes holds too few
     * values for four chains to cover the add latency on a core with several FMA pipes, which the measured
     * Apple and NEON hosts have; wider vectors already keep enough in flight with four.
     */
    private val NARROW = LANE <= 2

    /** [dotUnrolled] for [NARROW] vectors: eight chains, each updated once per iteration. */
    private fun dotEightChains(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        var s4 = DoubleVector.zero(SPECIES)
        var s5 = DoubleVector.zero(SPECIES)
        var s6 = DoubleVector.zero(SPECIES)
        var s7 = DoubleVector.zero(SPECIES)
        val unrolled = len - len % (8 * LANE)
        val aEnd = aOff + unrolled
        var aIndex = aOff
        var bIndex = bOff
        while (aIndex < aEnd) {
            s0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aIndex), DoubleVector.fromArray(SPECIES, b, bIndex), s0)
            s1 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + LANE),
                s1,
            )
            s2 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 2 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 2 * LANE),
                s2,
            )
            s3 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 3 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 3 * LANE),
                s3,
            )
            s4 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 4 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 4 * LANE),
                s4,
            )
            s5 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 5 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 5 * LANE),
                s5,
            )
            s6 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 6 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 6 * LANE),
                s6,
            )
            s7 = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aIndex + 7 * LANE),
                DoubleVector.fromArray(SPECIES, b, bIndex + 7 * LANE),
                s7,
            )
            aIndex += 8 * LANE
            bIndex += 8 * LANE
        }
        val head = s0.add(s1).add(s2.add(s3)).add(s4.add(s5).add(s6.add(s7))).reduceLanes(VectorOperators.ADD)
        var tail = 0.0
        val end = aOff + len
        while (aIndex < end) {
            tail += a[aIndex] * b[bIndex]
            aIndex++
            bIndex++
        }
        return head + tail
    }

    /** [asumUnrolled] for [NARROW] vectors, for the same reason as [dotEightChains]. */
    private fun asumEightChains(v: DoubleArray, vOff: Int, len: Int): Double {
        val mask = LongVector.fromArray(LONG_SPECIES, SIGN_MASK_BITS, 0)
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        var s4 = DoubleVector.zero(SPECIES)
        var s5 = DoubleVector.zero(SPECIES)
        var s6 = DoubleVector.zero(SPECIES)
        var s7 = DoubleVector.zero(SPECIES)
        val bound = vOff + len - len % (8 * LANE)
        var index = vOff
        while (index < bound) {
            s0 = s0.add(signStripped(DoubleVector.fromArray(SPECIES, v, index), mask))
            s1 = s1.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + LANE), mask))
            s2 = s2.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 2 * LANE), mask))
            s3 = s3.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 3 * LANE), mask))
            s4 = s4.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 4 * LANE), mask))
            s5 = s5.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 5 * LANE), mask))
            s6 = s6.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 6 * LANE), mask))
            s7 = s7.add(signStripped(DoubleVector.fromArray(SPECIES, v, index + 7 * LANE), mask))
            index += 8 * LANE
        }
        val head = s0.add(s1).add(s2.add(s3)).add(s4.add(s5).add(s6.add(s7))).reduceLanes(VectorOperators.ADD)
        var tail = 0.0
        val end = vOff + len
        while (index < end) {
            tail += abs(v[index])
            index++
        }
        return head + tail
    }
}
