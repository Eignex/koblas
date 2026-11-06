package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Shared conformance assertions cover owned kernels and Native host bindings.

/**
 * The level-1 kernels against the explicit scalar oracle, at a non-zero offset so an implementation that ignores
 * the offset fails.
 *
 * The lengths straddle every boundary an arm can have: a lane width, whatever unrolling a vectorised body
 * does, and the width at which the vendor arm crosses to the library. 600 clears an unrolled body on a
 * wide-lane host, whose own threshold scales with the lane count and so sits above the shorter lengths here.
 */
internal fun assertLevel1KernelsAgreeWithReference(kernels: KoblasEngine) {
    val rng = Random(20260731)
    for (len in intArrayOf(1, 7, 31, 32, 63, 64, 65, 200, 600)) {
        val pad = 3
        val a = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) }
        val b = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) }
        val expectedDot = ScalarEngine.dot(a, pad, b, 0, len)
        assertClose(
            doubleArrayOf(expectedDot),
            doubleArrayOf(kernels.dot(a, pad, b, 0, len)),
            context = "dot len=$len",
        )
        val expectedAxpy = b.copyOf()
        ScalarEngine.axpy(expectedAxpy, 0, 0.75, a, pad, len)
        val actualAxpy = b.copyOf()
        kernels.axpy(actualAxpy, 0, 0.75, a, pad, len)
        assertClose(expectedAxpy, actualAxpy, context = "axpy len=$len")
        val expectedScale = a.copyOf()
        ScalarEngine.scale(expectedScale, pad, -0.5, len)
        val actualScale = a.copyOf()
        kernels.scale(actualScale, pad, -0.5, len)
        assertClose(expectedScale, actualScale, context = "scale len=$len")
    }
}

/**
 * Plane rotation agreement with the portable kernel for every implementation, with the two operands at
 * different offsets so a kernel that addresses both through one offset fails.
 */
internal fun assertRotKernelAgreesWithReference(kernels: KoblasEngine) {
    // The rotation BLAS drotg(3, 4) produces, written out because koblas exposes no generator.
    val c = 0.6
    val s = 0.8
    val xOffset = 2
    val yOffset = 3
    for (len in intArrayOf(0, 1, 7, 63, 64, 65, 200)) {
        val x = DoubleArray(len + 2 * xOffset) { it * 0.25 - 4.0 }
        val y = DoubleArray(len + 2 * yOffset) { 3.0 - it * 0.125 }
        val expectedX = x.copyOf()
        val expectedY = y.copyOf()
        val actualX = x.copyOf()
        val actualY = y.copyOf()
        ScalarEngine.rot(expectedX, xOffset, expectedY, yOffset, len, c, s)
        kernels.rot(actualX, xOffset, actualY, yOffset, len, c, s)
        assertClose(expectedX, actualX, context = "rot x len=$len")
        assertClose(expectedY, actualY, context = "rot y len=$len")
    }
}

/**
 * Exchange, at a non-zero offset in both operands and at lengths that straddle a lane boundary, so a kernel
 * that swaps whole vectors is checked for leaving the padding either side untouched.
 */
internal fun assertSwapAgreesWithReference(kernels: KoblasEngine) {
    val rng = Random(20260826)
    for (len in intArrayOf(1, 7, 63, 64, 65, 200, 600)) {
        val pad = 3
        val a = DoubleArray(len + 2 * pad) { rng.nextDouble(-1.0, 1.0) }
        val b = DoubleArray(len + 2 * pad) { rng.nextDouble(-1.0, 1.0) }
        val expectedA = a.copyOf()
        val expectedB = b.copyOf()
        ScalarEngine.swap(expectedA, pad, expectedB, pad, len)

        kernels.swap(a, pad, b, pad, len)

        assertContentEquals(expectedA, a, "swap len=$len a")
        assertContentEquals(expectedB, b, "swap len=$len b")
    }
}

/**
 * The reductions at scales whose squares leave the exponent range, which is what forces `nrm2` to rescale
 * rather than sum squares directly, plus the zero run both must report as zero exactly.
 */
internal fun assertReductionsAgreeWithReference(kernels: KoblasEngine) {
    val rng = Random(20260951)
    for (scale in doubleArrayOf(1.0, 1e200, 1e-200)) {
        for (len in intArrayOf(1, 31, 32, 63, 64, 200, 600)) {
            val pad = 3
            val v = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) * scale }
            val ctx = "len=$len scale=$scale"
            assertClose(
                doubleArrayOf(ScalarEngine.nrm2(v, pad, len)),
                doubleArrayOf(kernels.nrm2(v, pad, len)),
                context = "nrm2 $ctx",
            )
            val expectedSum = ScalarEngine.sum(v, pad, len)
            assertClose(
                doubleArrayOf(expectedSum),
                doubleArrayOf(kernels.sum(v, pad, len)),
                context = "sum $ctx",
            )
            val expectedAsum = ScalarEngine.asum(v, pad, len)
            assertClose(
                doubleArrayOf(expectedAsum),
                doubleArrayOf(kernels.asum(v, pad, len)),
                context = "asum $ctx",
            )
        }
    }
    val zeros = DoubleArray(80)
    assertEquals(0.0, kernels.nrm2(zeros, 0, 80), "nrm2 of zeros")
    assertEquals(0.0, kernels.asum(zeros, 0, 80), "asum of zeros")
}

/**
 * Finite IAMAX inputs select an in-range maximum without fixing which tied position wins.
 *
 * The lengths straddle a lane block, the unrolled body at every lane count and the host crossovers up to
 * 512 entries.
 */
internal fun assertIamaxHonoursItsContract(kernels: KoblasEngine) {
    val rng = Random(20260912)
    for (len in intArrayOf(0, 1, 2, 7, 16, 17, 64, 255, 256, 257, 1024)) {
        for (off in intArrayOf(0, 1, 3)) {
            val inputs = listOf(
                DoubleArray(len) { rng.nextDouble(-1.0, 1.0) },
                DoubleArray(len) { it.toDouble() },
                DoubleArray(len) { (len - it).toDouble() },
                DoubleArray(len) { if (it % 2 == 0) -0.0 else 0.0 },
            )
            for ((fixture, input) in inputs.withIndex()) {
                // Padding is larger than anything inside, so reading past either end is visible here.
                val v = DoubleArray(off + len + 3) { Double.POSITIVE_INFINITY }
                input.copyInto(v, off)
                val context = "${kernels.name} len=$len off=$off fixture=$fixture"

                val index = kernels.iamax(v, off, len)

                if (len == 0) {
                    assertEquals(-1, index, context)
                    continue
                }
                assertTrue(index in 0 until len, "$context: $index is not a logical index")
                val chosen = abs(v[off + index])
                var best = 0.0
                for (i in 0 until len) best = maxOf(best, abs(input[i]))
                assertTrue(chosen == best, "$context: chose $chosen, largest was $best")
            }
        }
    }
}

/** Finite scaling accuracy and untouched padding, without fixing exceptional results or signed zeros. */
internal fun assertScaleAgreesWithReference(kernels: KoblasEngine) {
    forEachScaleFixture { off, len, alpha, expected, actual ->
        kernels.scale(actual, off, alpha, len)
        val context = "${kernels.name} len=$len off=$off alpha=$alpha"
        for (i in expected.indices) {
            if (i in off until off + len) {
                assertClose(expected[i], actual[i], "$context at $i")
            } else {
                assertEquals(expected[i], actual[i], "$context wrote outside its window at $i")
            }
        }
    }
}

/**
 * Finite inputs and multipliers at tail boundaries, with poisoned padding. The offset rotates with the
 * length, so every alignment is reached without running every length at each of them; 4097 is past the
 * host crossover for scaling.
 */
private inline fun forEachScaleFixture(body: (Int, Int, Double, DoubleArray, DoubleArray) -> Unit) {
    val finite = doubleArrayOf(0.0, -0.0, Double.MIN_VALUE, -Double.MIN_VALUE, 1e200, -1e200, 1e-200, -1e-200)
    val lengths = intArrayOf(
        0, 1, 3, 4, 7, 31, 32, 47, 48, 49, 63, 64, 65,
        127, 128, 129, 143, 144, 145, 159, 160, 161, 255, 256, 257, 4097,
    )
    for ((index, len) in lengths.withIndex()) {
        val off = index % 8
        for (alpha in doubleArrayOf(0.0, -0.0, 1.0, -1.0, 0.875)) {
            val expected = DoubleArray(off + len + 5) { if (it % 2 == 0) Double.NaN else Double.POSITIVE_INFINITY }
            for (i in 0 until len) {
                expected[off + i] =
                    if (i % 3 == 0) finite[(i / 3) % finite.size] else i * 0.125 - 3.0
            }
            val actual = expected.copyOf()
            ScalarEngine.scale(expected, off, alpha, len)
            body(off, len, alpha, expected, actual)
        }
    }
}

/** Norms whose components square outside the exponent range, compared with the scalar oracle. */
internal fun assertNormExtremesAgreeWithReference(kernels: KoblasEngine) {
    val pair = doubleArrayOf(3e200, 4e200)
    assertEquals(ScalarEngine.nrm2(pair, 0, 2), kernels.nrm2(pair, 0, 2), absoluteTolerance = 1e188)
    val tinyPair = doubleArrayOf(3e-200, 4e-200)
    assertEquals(ScalarEngine.nrm2(tinyPair, 0, 2), kernels.nrm2(tinyPair, 0, 2), absoluteTolerance = 1e-212)

    for (len in intArrayOf(16, 33, 64)) {
        val big = DoubleArray(len) { 1e200 }
        val expected = ScalarEngine.nrm2(big, 0, len)
        assertEquals(
            expected,
            kernels.nrm2(big, 0, len),
            absoluteTolerance = expected * 1e-12,
        )
        val tiny = DoubleArray(len) { 1e-200 }
        val expectedTiny = ScalarEngine.nrm2(tiny, 0, len)
        assertEquals(
            expectedTiny,
            kernels.nrm2(tiny, 0, len),
            absoluteTolerance = expectedTiny * 1e-12,
        )
    }
}

/** Offset windows and empty tails compared with the scalar oracle. */
internal fun assertNormWindowsAgreeWithReference(kernels: KoblasEngine) {
    val rng = Random(20260815)
    val v = DoubleArray(300) { rng.nextDouble(-1.0, 1.0) }
    for (off in intArrayOf(0, 1, 7)) {
        for (len in intArrayOf(0, 1, 3, 8, 31, 128, 293)) {
            val expected = ScalarEngine.nrm2(v, off, len)
            assertEquals(
                expected,
                kernels.nrm2(v, off, len),
                absoluteTolerance = 1e-12 * (expected + 1.0),
                message = "off $off len $len",
            )
        }
    }
}
