package com.eignex.koblas

import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The level-1 seam: that a registered [Level1] is reached for long runs, is not reached for short ones,
 * and that ranking and overriding behave like the other two halves.
 *
 * This lives in `commonTest` because the seam itself now lives in `commonMain`. It could not before: the
 * interface and its install hook were declared in a non-JVM source set, so the mechanism was only
 * testable on the targets that happened to have a host library, and the JVM could not see it at all.
 *
 * The routing assertions are written against [DispatchThresholds.level1] rather than a hardcoded length,
 * because the threshold is a per-platform value and is [Int.MAX_VALUE] on a SIMD JVM — where the correct
 * behaviour is that nothing routes at any length.
 */
class Level1Test {

    /** Records what it was asked to do and delegates the arithmetic to a plain loop. */
    private class Recording(override val priority: Int = 90) : Level1 {
        override val name: String get() = "recording"
        var dots = 0
        var axpys = 0
        var scales = 0

        override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
            dots++
            var s = 0.0
            for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
            return s
        }

        override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
            axpys++
            for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
        }

        override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
            scales++
            for (i in 0 until len) v[vOff + i] *= alpha
        }

        var nrm2s = 0
        var asums = 0

        override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double {
            nrm2s++
            var s = 0.0
            for (i in 0 until len) s += v[vOff + i] * v[vOff + i]
            return kotlin.math.sqrt(s)
        }

        override fun asum(v: DoubleArray, vOff: Int, len: Int): Double {
            asums++
            var s = 0.0
            for (i in 0 until len) s += kotlin.math.abs(v[vOff + i])
            return s
        }
    }

    @AfterTest
    fun restore() {
        resetRegisteredLevel1()
        registerPlatformBackends()
    }

    @Test
    fun `a registered backend is reached exactly above the level-1 threshold`() {
        resetRegisteredLevel1()
        val recording = Recording()
        registerLevel1(recording)
        val threshold = dispatchThresholds.level1
        val long = if (threshold == Int.MAX_VALUE) 4096 else threshold
        val x = DenseVector.of(DoubleArray(long) { 1.0 })
        val y = DenseVector.of(DoubleArray(long) { 2.0 })
        assertEquals(2.0 * long, x dot y, "the arithmetic must be right whichever path ran")
        if (threshold == Int.MAX_VALUE) {
            assertEquals(0, recording.dots, "nothing should route when the platform keeps level 1 portable")
        } else {
            assertEquals(1, recording.dots, "a run at the threshold should route")
        }
        // One element is below every threshold this library ships, so it must never route.
        val before = recording.dots
        val short1 = DenseVector.of(doubleArrayOf(3.0))
        val short2 = DenseVector.of(doubleArrayOf(4.0))
        assertEquals(12.0, short1 dot short2)
        assertEquals(before, recording.dots, "a length-1 dot routed to the backend")
    }

    @Test
    fun `axpy and scale route on the same threshold as dot`() {
        resetRegisteredLevel1()
        val recording = Recording()
        registerLevel1(recording)
        val threshold = dispatchThresholds.level1
        if (threshold == Int.MAX_VALUE) return
        val v = DenseVector.of(DoubleArray(threshold) { 1.0 })
        val x = DenseVector.of(DoubleArray(threshold) { 2.0 })
        axpy(v, 3.0, x)
        scale(v, 0.5)
        assertEquals(1, recording.axpys, "axpy did not route")
        assertEquals(1, recording.scales, "scale did not route")
        assertTrue(v.data.all { it == 3.5 }, "routed arithmetic is wrong: ${v.data[0]}")
    }

    @Test
    fun `the dense reductions route but the sparse ones cannot`() {
        resetRegisteredLevel1()
        val recording = Recording()
        registerLevel1(recording)
        val threshold = dispatchThresholds.level1
        if (threshold == Int.MAX_VALUE) return
        val dense = DenseVector.of(DoubleArray(threshold) { 3.0 })
        norm2(dense)
        asum(dense)
        assertEquals(1, recording.nrm2s, "norm2 on a dense vector did not route")
        assertEquals(1, recording.asums, "asum on a dense vector did not route")
        // A sparse vector has no BLAS counterpart: it must stay on the stored-entry walk.
        val sparse = SparseVector(threshold, IntArray(threshold) { it }, DoubleArray(threshold) { 3.0 })
        norm2(sparse)
        asum(sparse)
        assertEquals(1, recording.nrm2s, "norm2 routed a sparse vector")
        assertEquals(1, recording.asums, "asum routed a sparse vector")
    }

    /**
     * iamax stays off the seam by design; see [Level1]. Its tie-breaking and NaN ranking are koblas's own
     * contract, and `idamax` implementations disagree about the latter, so routing it would make the answer
     * depend on whether a host library happened to be installed.
     */
    @Test
    fun `iamax does not route`() {
        resetRegisteredLevel1()
        val recording = Recording()
        registerLevel1(recording)
        val threshold = dispatchThresholds.level1
        val len = if (threshold == Int.MAX_VALUE) 4096 else threshold
        val v = DenseVector.of(DoubleArray(len) { if (it == 7) -9.0 else 1.0 })
        assertEquals(7, iamax(v))
        assertEquals(0, recording.dots + recording.nrm2s + recording.asums, "iamax reached the seam")
    }

    @Test
    fun `registration keeps the highest priority and install overrides both`() {
        resetRegisteredLevel1()
        val weak = Recording(priority = 10)
        val strong = Recording(priority = 200)
        registerLevel1(strong)
        registerLevel1(weak)
        assertEquals(strong, activeLevel1, "a weaker registration displaced a stronger one")
        val override = Recording(priority = 0)
        installLevel1(override)
        assertEquals(override, activeLevel1, "install must win regardless of priority")
        installLevel1(null)
        assertEquals(strong, activeLevel1, "clearing the override should fall back to registration")
        resetRegisteredLevel1()
        assertEquals(null, activeLevel1, "a cleared registry must fall back to the built-in kernels")
    }
}
