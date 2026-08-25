package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.assertEquals

// The F64Kernels contract, over any implementation. The compiled-in kernels and a host binding must both
// satisfy it, and a host kernels class exists only on the native targets, so the assertions live here rather
// than beside either caller.

/**
 * The level-1 kernels against loops written out here, at a non-zero offset so an implementation that ignores
 * the offset fails. Lengths 63, 64 and 65 straddle the level-1 routing threshold, so both the dispatched and
 * the direct path are covered. 600 clears any unrolled body a wide-lane host takes, whose own threshold
 * scales with the lane count and so sits above the shorter lengths here.
 */
internal fun assertLevel1KernelsAgreeWithScalar(kernels: F64Kernels) {
    val rng = Random(20260731)
    for (len in intArrayOf(1, 7, 63, 64, 65, 200, 600)) {
        val pad = 3
        val a = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) }
        val b = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) }
        var expectedDot = 0.0
        for (i in 0 until len) expectedDot += a[pad + i] * b[i]
        assertClose(
            doubleArrayOf(expectedDot),
            doubleArrayOf(kernels.dot(a, pad, b, 0, len)),
            context = "dot len=$len",
        )
        val expectedAxpy = b.copyOf()
        for (i in 0 until len) expectedAxpy[i] += 0.75 * a[pad + i]
        val actualAxpy = b.copyOf()
        kernels.axpy(actualAxpy, 0, 0.75, a, pad, len)
        assertClose(expectedAxpy, actualAxpy, context = "axpy len=$len")
        val expectedScale = a.copyOf()
        for (i in 0 until len) expectedScale[pad + i] *= -0.5
        val actualScale = a.copyOf()
        kernels.scale(actualScale, pad, -0.5, len)
        assertClose(expectedScale, actualScale, context = "scale len=$len")
    }
}

/**
 * Exchange, at a non-zero offset in both operands and at lengths that straddle a lane boundary, so a kernel
 * that swaps whole vectors is checked for leaving the padding either side untouched.
 */
internal fun assertSwapAgreesWithScalar(kernels: F64Kernels) {
    val rng = Random(20260826)
    for (len in intArrayOf(1, 7, 63, 64, 65, 200, 600)) {
        val pad = 3
        val a = DoubleArray(len + 2 * pad) { rng.nextDouble(-1.0, 1.0) }
        val b = DoubleArray(len + 2 * pad) { rng.nextDouble(-1.0, 1.0) }
        val aBefore = a.copyOf()
        val bBefore = b.copyOf()
        kernels.swap(a, pad, b, pad, len)
        for (i in 0 until len) {
            assertEquals(bBefore[pad + i], a[pad + i], "swap len=$len a($i)")
            assertEquals(aBefore[pad + i], b[pad + i], "swap len=$len b($i)")
        }
        // Everything outside the run has to survive, which is what an overrunning vector store would break.
        for (i in 0 until pad) {
            assertEquals(aBefore[i], a[i], "swap len=$len wrote before a")
            assertEquals(bBefore[i], b[i], "swap len=$len wrote before b")
            assertEquals(aBefore[pad + len + i], a[pad + len + i], "swap len=$len wrote past a")
            assertEquals(bBefore[pad + len + i], b[pad + len + i], "swap len=$len wrote past b")
        }
    }
}

/**
 * The reductions at scales whose squares leave the exponent range, which is what forces `nrm2` to rescale
 * rather than sum squares directly, plus the zero run both must report as zero exactly.
 */
internal fun assertReductionsAgreeWithScalar(kernels: F64Kernels) {
    val rng = Random(20260951)
    for (scale in doubleArrayOf(1.0, 1e200, 1e-200)) {
        for (len in intArrayOf(1, 63, 64, 200, 600)) {
            val pad = 3
            val v = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) * scale }
            val ctx = "len=$len scale=$scale"
            assertClose(
                doubleArrayOf(rescaledNorm(v, pad, len)),
                doubleArrayOf(kernels.nrm2(v, pad, len)),
                context = "nrm2 $ctx",
            )
            var expectedAsum = 0.0
            for (i in 0 until len) expectedAsum += abs(v[pad + i])
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

/** The rescaled two-pass norm, written out so the oracle does not use the implementation under test. */
private fun rescaledNorm(v: DoubleArray, off: Int, len: Int): Double {
    var amax = 0.0
    for (i in 0 until len) {
        val a = abs(v[off + i])
        if (a > amax) amax = a
    }
    if (amax == 0.0) return 0.0
    var sum = 0.0
    for (i in 0 until len) {
        val scaled = v[off + i] / amax
        sum += scaled * scaled
    }
    return amax * sqrt(sum)
}
