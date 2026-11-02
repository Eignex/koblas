@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.dense.PortableTriangularKernels
import com.eignex.koblas.internal.dense.triangleEntry
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/**
 * JVM Vector API diagonal substitution with portable fallbacks.
 *
 * Lanes cover independent right-hand sides; substitution steps remain a dependency chain.
 * Coefficients and diagonals are broadcast across the sides.
 *
 * Use division directly: a reciprocal of a subnormal diagonal can overflow even when the quotient
 * is finite. Zero and infinite diagonals also retain division semantics.
 *
 * Strided sides and incomplete lane blocks use [PortableTriangularKernels];
 * [implementationsFor] reports those bodies. [gathersRightHandSides] requests a copy when it can
 * amortize vector loads. Species initialization requires the Vector API module, checked by
 * [BuiltinEngines] before using this backend.
 */
internal object SimdTriangularKernels : KoblasEngine() {
    override val id: String = "jvm-simd"
    override val implementations get() = setOf(EngineImplementation.Triangular)

    private val SPECIES = SIMD_SPECIES
    private val LANE = SPECIES.length()

    /** Built once, for the reason [SimdPanelKernels] builds its own once. */
    private val NAME: String = "simd-substitution($LANE lanes)"

    override val name: String get() = NAME

    /**
     * Group right-hand sides so the block repeatedly read during substitution stays resident.
     * This cache grouping is independent of the lane count.
     */
    private const val GROUP_BLOCKS = 4

    /** Steps a single-side substitution finishes together before one vector pass carries them on. */
    private const val GROUP = 4

    /** Lane blocks of sides a diagonal substitution carries through each step together. */
    private const val FOUR_BLOCKS = 4

    override fun rightHandSideGroup(order: Int, sides: Int): Int {
        val recommended = GROUP_BLOCKS * LANE
        if (sides in 1 until recommended) return sides
        return if (sides < 1) 1 else recommended
    }

    /**
     * Gather strided sides when they fill a lane block and the diagonal block has multiple steps.
     * The copy is paid once and reused through the block; fewer sides reach no vector body, and a
     * single step offers no reuse.
     */
    override fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean =
        rhsStride != 1 && sides >= LANE && order > 1

    override fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> = when {
        order <= 0 || sides <= 0 -> emptyList()
        sides == 1 && contiguous -> listOf(NAME)
        !contiguous || sides < LANE -> PortableTriangularKernels.implementationsFor(order, sides, contiguous)
        sides % LANE == 0 -> listOf(NAME)
        else -> listOf(NAME) + PortableTriangularKernels.implementationsFor(order, sides, contiguous)
    }

    override fun diagonalSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int,
    ) {
        if (size <= 0 || sides <= 0) return
        if (sides == 1 && orderStride == 1) {
            singleSolve(t, n, start, size, transposed, lower, unitDiag, x, xOffset, tOffset)
            return
        }
        if (rhsStride != 1 || sides < LANE) {
            PortableTriangularKernels.diagonalSolve(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
            )
            return
        }
        var lane = 0
        val bound = sides - sides % LANE
        // Unfused four-block compilation exceeds the JIT budget and materializes vectors on the heap.
        while (hardwareFusedMultiplyAdd && lane + FOUR_BLOCKS * LANE <= bound) {
            solveFour(t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride, tOffset)
            lane += FOUR_BLOCKS * LANE
        }
        while (lane < bound) {
            solveBlock(t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride, tOffset)
            lane += LANE
        }
        if (lane < sides) {
            PortableTriangularKernels.diagonalSolve(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride,
                1, sides - lane, tOffset,
            )
        }
    }

    override fun diagonalMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int,
    ) {
        if (size <= 0 || sides <= 0) return
        if (sides == 1 && orderStride == 1) {
            singleMultiply(t, n, start, size, transposed, lower, unitDiag, x, xOffset, tOffset)
            return
        }
        if (rhsStride != 1 || sides < LANE) {
            PortableTriangularKernels.diagonalMultiply(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
            )
            return
        }
        var lane = 0
        val bound = sides - sides % LANE
        // Unfused four-block compilation exceeds the JIT budget and materializes vectors on the heap.
        while (hardwareFusedMultiplyAdd && lane + FOUR_BLOCKS * LANE <= bound) {
            multiplyFour(t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride, tOffset)
            lane += FOUR_BLOCKS * LANE
        }
        while (lane < bound) {
            multiplyBlock(t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride, tOffset)
            lane += LANE
        }
        if (lane < sides) {
            PortableTriangularKernels.diagonalMultiply(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride,
                1, sides - lane, tOffset,
            )
        }
    }

    /**
     * Walk a whole diagonal block per lane block of sides to keep repeatedly read values resident.
     * Interchanging the loops would sweep all sides between successive uses.
     */
    private fun solveBlock(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        tOffset: Int,
    ) {
        for (step in 0 until size) {
            val p = if (lower) step else size - 1 - step
            val at = xOffset + p * orderStride
            var accumulated = DoubleVector.fromArray(SPECIES, x, at)
            val first = if (lower) 0 else p + 1
            val last = if (lower) p else size
            for (q in first until last) {
                val coefficient = triangleEntry(t, n, start, p, q, transposed, tOffset)
                accumulated = multiplySubtract(
                    DoubleVector.fromArray(SPECIES, x, xOffset + q * orderStride),
                    coefficient,
                    accumulated,
                )
            }
            if (!unitDiag) accumulated = accumulated.div(triangleEntry(t, n, start, p, p, transposed, tOffset))
            accumulated.intoArray(x, at)
        }
    }

    /**
     * [solveBlock] over four adjacent lane blocks at once.
     *
     * One lane block is a single chain of fused steps, each waiting on the last, so it runs at the
     * instruction's latency rather than its throughput. Four blocks are four independent chains that share
     * each coefficient's broadcast.
     */
    private fun solveFour(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        tOffset: Int,
    ) {
        for (step in 0 until size) {
            val p = if (lower) step else size - 1 - step
            val at = xOffset + p * orderStride
            var a0 = DoubleVector.fromArray(SPECIES, x, at)
            var a1 = DoubleVector.fromArray(SPECIES, x, at + LANE)
            var a2 = DoubleVector.fromArray(SPECIES, x, at + 2 * LANE)
            var a3 = DoubleVector.fromArray(SPECIES, x, at + 3 * LANE)
            val first = if (lower) 0 else p + 1
            val last = if (lower) p else size
            for (q in first until last) {
                val entry = triangleEntry(t, n, start, p, q, transposed, tOffset)
                val coefficient = DoubleVector.broadcast(SPECIES, -entry)
                val from = xOffset + q * orderStride
                a0 = fusedStep(DoubleVector.fromArray(SPECIES, x, from), coefficient, a0)
                a1 = fusedStep(DoubleVector.fromArray(SPECIES, x, from + LANE), coefficient, a1)
                a2 = fusedStep(DoubleVector.fromArray(SPECIES, x, from + 2 * LANE), coefficient, a2)
                a3 = fusedStep(DoubleVector.fromArray(SPECIES, x, from + 3 * LANE), coefficient, a3)
            }
            if (!unitDiag) {
                val diagonal = triangleEntry(t, n, start, p, p, transposed, tOffset)
                a0 = a0.div(diagonal)
                a1 = a1.div(diagonal)
                a2 = a2.div(diagonal)
                a3 = a3.div(diagonal)
            }
            a0.intoArray(x, at)
            a1.intoArray(x, at + LANE)
            a2.intoArray(x, at + 2 * LANE)
            a3.intoArray(x, at + 3 * LANE)
        }
    }

    /** [multiplyBlock] over four adjacent lane blocks at once, for the reason [solveFour] gives. */
    private fun multiplyFour(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        tOffset: Int,
    ) {
        for (step in 0 until size) {
            val p = if (lower) size - 1 - step else step
            val at = xOffset + p * orderStride
            var a0 = DoubleVector.fromArray(SPECIES, x, at)
            var a1 = DoubleVector.fromArray(SPECIES, x, at + LANE)
            var a2 = DoubleVector.fromArray(SPECIES, x, at + 2 * LANE)
            var a3 = DoubleVector.fromArray(SPECIES, x, at + 3 * LANE)
            if (!unitDiag) {
                val diagonal = triangleEntry(t, n, start, p, p, transposed, tOffset)
                a0 = a0.mul(diagonal)
                a1 = a1.mul(diagonal)
                a2 = a2.mul(diagonal)
                a3 = a3.mul(diagonal)
            }
            val first = if (lower) 0 else p + 1
            val last = if (lower) p else size
            for (q in first until last) {
                val entry = triangleEntry(t, n, start, p, q, transposed, tOffset)
                val coefficient = DoubleVector.broadcast(SPECIES, entry)
                val from = xOffset + q * orderStride
                a0 = fusedStep(DoubleVector.fromArray(SPECIES, x, from), coefficient, a0)
                a1 = fusedStep(DoubleVector.fromArray(SPECIES, x, from + LANE), coefficient, a1)
                a2 = fusedStep(DoubleVector.fromArray(SPECIES, x, from + 2 * LANE), coefficient, a2)
                a3 = fusedStep(DoubleVector.fromArray(SPECIES, x, from + 3 * LANE), coefficient, a3)
            }
            a0.intoArray(x, at)
            a1.intoArray(x, at + LANE)
            a2.intoArray(x, at + 2 * LANE)
            a3.intoArray(x, at + 3 * LANE)
        }
    }

    /** The product counterpart of [solveBlock], over the steps that still hold what they arrived with. */
    private fun multiplyBlock(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        tOffset: Int,
    ) {
        for (step in 0 until size) {
            val p = if (lower) size - 1 - step else step
            val at = xOffset + p * orderStride
            val own = DoubleVector.fromArray(SPECIES, x, at)
            var accumulated =
                if (unitDiag) own else own.mul(triangleEntry(t, n, start, p, p, transposed, tOffset))
            val first = if (lower) 0 else p + 1
            val last = if (lower) p else size
            for (q in first until last) {
                val coefficient = triangleEntry(t, n, start, p, q, transposed, tOffset)
                accumulated = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, x, xOffset + q * orderStride),
                    coefficient,
                    accumulated,
                )
            }
            accumulated.intoArray(x, at)
        }
    }

    /**
     * One contiguous right-hand side, vectorised along the block instead of across sides.
     *
     * Four steps finish together: their corner is substituted in scalars, and then one vector pass carries all
     * four into the steps that remain, which untransposed are the columns' runs below them and transposed are
     * the rows' dots against the steps before them. A step at a time paid a pass and a scalar tail per step,
     * which on a block a few lane blocks long cost as much as the arithmetic.
     *
     * Line `k` of the block, a column untransposed and a row transposed, starts at `base + k · n`, so both
     * orientations walk the same memory and differ only in whether a pass reads a line or reduces it.
     */
    private fun singleSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        tOffset: Int,
    ) {
        val base = tOffset + start + start * n
        val direction = if (lower) 1 else -1
        var done = 0
        while (size - done >= GROUP) {
            val p0 = if (lower) done else size - 1 - done
            val p1 = p0 + direction
            val p2 = p0 + 2 * direction
            val p3 = p0 + 3 * direction
            if (transposed) {
                // Reduce each row against the steps already finished, which lie before the group.
                val from = if (lower) 0 else p0 + 1
                val until = if (lower) p0 else size
                fourDots(t, base, n, p0, p1, p2, p3, x, xOffset, from, until - from, -1.0)
            }
            val x0 = divide(x[xOffset + p0], t, base, n, p0, unitDiag)
            val x1 = divide(x[xOffset + p1] - entry(t, base, n, transposed, p1, p0) * x0, t, base, n, p1, unitDiag)
            val x2 = divide(
                x[xOffset + p2] - entry(
                    t,
                    base,
                    n,
                    transposed,
                    p2,
                    p0,
                ) * x0 - entry(t, base, n, transposed, p2, p1) * x1,
                t,
                base,
                n,
                p2,
                unitDiag,
            )
            val x3 = divide(
                x[xOffset + p3] - entry(t, base, n, transposed, p3, p0) * x0 -
                    entry(t, base, n, transposed, p3, p1) * x1 - entry(t, base, n, transposed, p3, p2) * x2,
                t,
                base,
                n,
                p3,
                unitDiag,
            )
            x[xOffset + p0] = x0
            x[xOffset + p1] = x1
            x[xOffset + p2] = x2
            x[xOffset + p3] = x3
            if (!transposed) {
                // Carry the four finished steps into the ones after the group.
                val from = if (lower) p3 + 1 else 0
                val until = if (lower) size else p3
                fourUpdate(t, base, n, p0, p1, p2, p3, x, xOffset, from, until - from, -x0, -x1, -x2, -x3)
            }
            done += GROUP
        }
        for (step in done until size) {
            val p = if (lower) step else size - 1 - step
            if (transposed) {
                val from = if (lower) 0 else p + 1
                val until = if (lower) p else size
                x[xOffset + p] -= runDot(t, base + p * n + from, x, xOffset + from, until - from)
                x[xOffset + p] = divide(x[xOffset + p], t, base, n, p, unitDiag)
            } else {
                val value = divide(x[xOffset + p], t, base, n, p, unitDiag)
                x[xOffset + p] = value
                val from = if (lower) p + 1 else 0
                val until = if (lower) size else p
                runUpdate(t, base + p * n + from, x, xOffset + from, until - from, -value)
            }
        }
    }

    /**
     * The multiply counterpart of [singleSolve]. Groups run from the end a step's own value is needed last,
     * and each finished step is `M(p, p) · x(p)` plus the steps after it in that order, inside the group from
     * the corner and outside it from one vector pass over inputs no group has overwritten yet.
     */
    private fun singleMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        tOffset: Int,
    ) {
        val base = tOffset + start + start * n
        val direction = if (lower) -1 else 1
        var done = 0
        while (size - done >= GROUP) {
            val p0 = if (lower) size - 1 - done else done
            val p1 = p0 + direction
            val p2 = p0 + 2 * direction
            val p3 = p0 + 3 * direction
            val x0 = x[xOffset + p0]
            val x1 = x[xOffset + p1]
            val x2 = x[xOffset + p2]
            val x3 = x[xOffset + p3]
            if (!transposed) {
                // The group's columns reach the steps already finished, which lie before it in this order.
                val from = if (lower) p0 + 1 else 0
                val until = if (lower) size else p0
                fourUpdate(t, base, n, p0, p1, p2, p3, x, xOffset, from, until - from, x0, x1, x2, x3)
            }
            x[xOffset + p0] = scale(x0, t, base, n, p0, unitDiag) + entry(t, base, n, transposed, p0, p1) * x1 +
                entry(t, base, n, transposed, p0, p2) * x2 + entry(t, base, n, transposed, p0, p3) * x3
            x[xOffset + p1] = scale(x1, t, base, n, p1, unitDiag) + entry(t, base, n, transposed, p1, p2) * x2 +
                entry(t, base, n, transposed, p1, p3) * x3
            x[xOffset + p2] = scale(x2, t, base, n, p2, unitDiag) + entry(t, base, n, transposed, p2, p3) * x3
            x[xOffset + p3] = scale(x3, t, base, n, p3, unitDiag)
            if (transposed) {
                // Each row also reduces the steps not yet reached, which still hold their inputs.
                val from = if (lower) 0 else p3 + 1
                val until = if (lower) p3 else size
                fourDots(t, base, n, p0, p1, p2, p3, x, xOffset, from, until - from, 1.0)
            }
            done += GROUP
        }
        for (step in done until size) {
            val p = if (lower) size - 1 - step else step
            val own = x[xOffset + p]
            val scaled = scale(own, t, base, n, p, unitDiag)
            if (transposed) {
                val from = if (lower) 0 else p + 1
                val until = if (lower) p else size
                x[xOffset + p] = scaled + runDot(t, base + p * n + from, x, xOffset + from, until - from)
            } else {
                x[xOffset + p] = scaled
                val from = if (lower) p + 1 else 0
                val until = if (lower) size else p
                runUpdate(t, base + p * n + from, x, xOffset + from, until - from, own)
            }
        }
    }

    /** `M(i, j)`: line `j` at offset `i` untransposed, line `i` at offset `j` transposed. */
    private fun entry(t: DoubleArray, base: Int, n: Int, transposed: Boolean, i: Int, j: Int): Double =
        if (transposed) t[base + i * n + j] else t[base + j * n + i]

    private fun divide(value: Double, t: DoubleArray, base: Int, n: Int, p: Int, unitDiag: Boolean): Double =
        if (unitDiag) value else value / t[base + p * n + p]

    private fun scale(value: Double, t: DoubleArray, base: Int, n: Int, p: Int, unitDiag: Boolean): Double =
        if (unitDiag) value else t[base + p * n + p] * value

    /** `x(q) += s0 · line(p0, q) + … + s3 · line(p3, q)` over one run of steps, the column pass of a group. */
    private fun fourUpdate(
        t: DoubleArray,
        base: Int,
        n: Int,
        p0: Int,
        p1: Int,
        p2: Int,
        p3: Int,
        x: DoubleArray,
        xOffset: Int,
        from: Int,
        length: Int,
        s0: Double,
        s1: Double,
        s2: Double,
        s3: Double,
    ) {
        val l0 = base + p0 * n + from
        val l1 = base + p1 * n + from
        val l2 = base + p2 * n + from
        val l3 = base + p3 * n + from
        val at = xOffset + from
        var i = 0
        val bound = SPECIES.loopBound(length)
        if (bound > 0) {
            val c0 = DoubleVector.broadcast(SPECIES, s0)
            val c1 = DoubleVector.broadcast(SPECIES, s1)
            val c2 = DoubleVector.broadcast(SPECIES, s2)
            val c3 = DoubleVector.broadcast(SPECIES, s3)
            while (i < bound) {
                var sum = DoubleVector.fromArray(SPECIES, x, at + i)
                sum = multiplyAdd(DoubleVector.fromArray(SPECIES, t, l0 + i), c0, sum)
                sum = multiplyAdd(DoubleVector.fromArray(SPECIES, t, l1 + i), c1, sum)
                sum = multiplyAdd(DoubleVector.fromArray(SPECIES, t, l2 + i), c2, sum)
                multiplyAdd(DoubleVector.fromArray(SPECIES, t, l3 + i), c3, sum).intoArray(x, at + i)
                i += LANE
            }
        }
        while (i < length) {
            x[at + i] += s0 * t[l0 + i] + s1 * t[l1 + i] + s2 * t[l2 + i] + s3 * t[l3 + i]
            i++
        }
    }

    /** `x(pk) += sign · (line(pk) · x)` over one run of steps for four lines at once, the row pass of a group. */
    private fun fourDots(
        t: DoubleArray,
        base: Int,
        n: Int,
        p0: Int,
        p1: Int,
        p2: Int,
        p3: Int,
        x: DoubleArray,
        xOffset: Int,
        from: Int,
        length: Int,
        sign: Double,
    ) {
        val l0 = base + p0 * n + from
        val l1 = base + p1 * n + from
        val l2 = base + p2 * n + from
        val l3 = base + p3 * n + from
        val at = xOffset + from
        var i = 0
        var d0 = 0.0
        var d1 = 0.0
        var d2 = 0.0
        var d3 = 0.0
        val bound = SPECIES.loopBound(length)
        if (bound > 0) {
            var v0 = DoubleVector.zero(SPECIES)
            var v1 = DoubleVector.zero(SPECIES)
            var v2 = DoubleVector.zero(SPECIES)
            var v3 = DoubleVector.zero(SPECIES)
            while (i < bound) {
                val xv = DoubleVector.fromArray(SPECIES, x, at + i)
                v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, t, l0 + i), xv, v0)
                v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, t, l1 + i), xv, v1)
                v2 = multiplyAdd(DoubleVector.fromArray(SPECIES, t, l2 + i), xv, v2)
                v3 = multiplyAdd(DoubleVector.fromArray(SPECIES, t, l3 + i), xv, v3)
                i += LANE
            }
            d0 = v0.reduceLanes(VectorOperators.ADD)
            d1 = v1.reduceLanes(VectorOperators.ADD)
            d2 = v2.reduceLanes(VectorOperators.ADD)
            d3 = v3.reduceLanes(VectorOperators.ADD)
        }
        while (i < length) {
            val xi = x[at + i]
            d0 += t[l0 + i] * xi
            d1 += t[l1 + i] * xi
            d2 += t[l2 + i] * xi
            d3 += t[l3 + i] * xi
            i++
        }
        x[xOffset + p0] += sign * d0
        x[xOffset + p1] += sign * d1
        x[xOffset + p2] += sign * d2
        x[xOffset + p3] += sign * d3
    }

    /** `x(i) += scale · t(i)` along one contiguous run, the column step of [singleSolve]. */
    private fun runUpdate(t: DoubleArray, tFrom: Int, x: DoubleArray, xFrom: Int, length: Int, scale: Double) {
        var i = 0
        val bound = SPECIES.loopBound(length)
        if (bound > 0) {
            val coefficient = DoubleVector.broadcast(SPECIES, scale)
            while (i < bound) {
                multiplyAdd(
                    DoubleVector.fromArray(SPECIES, t, tFrom + i),
                    coefficient,
                    DoubleVector.fromArray(SPECIES, x, xFrom + i),
                ).intoArray(x, xFrom + i)
                i += LANE
            }
        }
        while (i < length) {
            x[xFrom + i] += scale * t[tFrom + i]
            i++
        }
    }

    /** The dot of two contiguous runs, the row step of [singleSolve], with two chains on a long run. */
    private fun runDot(t: DoubleArray, tFrom: Int, x: DoubleArray, xFrom: Int, length: Int): Double {
        var i = 0
        var sum = 0.0
        val bound = SPECIES.loopBound(length)
        if (bound > 0) {
            var even = DoubleVector.zero(SPECIES)
            var odd = DoubleVector.zero(SPECIES)
            while (i + 2 * LANE <= bound) {
                even = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, t, tFrom + i),
                    DoubleVector.fromArray(SPECIES, x, xFrom + i),
                    even,
                )
                odd = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, t, tFrom + i + LANE),
                    DoubleVector.fromArray(SPECIES, x, xFrom + i + LANE),
                    odd,
                )
                i += 2 * LANE
            }
            if (i < bound) {
                even = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, t, tFrom + i),
                    DoubleVector.fromArray(SPECIES, x, xFrom + i),
                    even,
                )
                i += LANE
            }
            sum = even.add(odd).reduceLanes(VectorOperators.ADD)
        }
        while (i < length) {
            sum += t[tFrom + i] * x[xFrom + i]
            i++
        }
        return sum
    }

    /**
     * `accumulated - coefficient · values`, fused where the machine has the instruction.
     *
     * Inline because a helper returning a [DoubleVector] that the JIT declines to inline makes the vector a
     * heap object, which would turn the substitution's one live accumulator into an allocation per step of
     * the block. The negated coefficient is what lets the fused form carry the subtraction.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun multiplySubtract(
        values: DoubleVector,
        coefficient: Double,
        accumulated: DoubleVector,
    ): DoubleVector = if (hardwareFusedMultiplyAdd) {
        values.fma(DoubleVector.broadcast(SPECIES, -coefficient), accumulated)
    } else {
        accumulated.sub(values.mul(coefficient))
    }

    /** `accumulated + coefficient · values`, inline for the reason [multiplySubtract] is. */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun multiplyAdd(
        values: DoubleVector,
        coefficient: Double,
        accumulated: DoubleVector,
    ): DoubleVector = if (hardwareFusedMultiplyAdd) {
        values.fma(DoubleVector.broadcast(SPECIES, coefficient), accumulated)
    } else {
        accumulated.add(values.mul(coefficient))
    }

    /** `accumulated + coefficient · values` over a broadcast coefficient, inline as [multiplySubtract] is. */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun fusedStep(
        values: DoubleVector,
        coefficient: DoubleVector,
        accumulated: DoubleVector,
    ): DoubleVector = if (hardwareFusedMultiplyAdd) {
        values.fma(coefficient, accumulated)
    } else {
        accumulated.add(values.mul(coefficient))
    }
}
