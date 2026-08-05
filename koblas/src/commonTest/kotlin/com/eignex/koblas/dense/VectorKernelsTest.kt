package com.eignex.koblas.dense

import com.eignex.koblas.DenseVector
import com.eignex.koblas.DispatchThresholds
import com.eignex.koblas.SparseVector
import com.eignex.koblas.asum
import com.eignex.koblas.axpy
import com.eignex.koblas.dispatchThresholds
import com.eignex.koblas.dot
import com.eignex.koblas.iamax
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.mathBackend
import com.eignex.koblas.norm2
import com.eignex.koblas.registerBackend
import com.eignex.koblas.resetBackends
import com.eignex.koblas.scale
import com.eignex.koblas.withCleanBackends
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * The level-1 seam: that a registered [VectorKernels] is reached for long runs, is not reached for short ones,
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
class VectorKernelsTest {

    /** Records what it was asked to do and delegates the arithmetic to a plain loop. */
    private class Recording(override val priority: Int = 90) : VectorKernels {
        override var name: String = "recording"
            private set

        /** Names this instance, so a routed `platform+name` assertion can tell two of them apart. */
        fun named(n: String): Recording = also { it.name = n }
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
            return sqrt(s)
        }

        override fun asum(v: DoubleArray, vOff: Int, len: Int): Double {
            asums++
            var s = 0.0
            for (i in 0 until len) s += abs(v[vOff + i])
            return s
        }
    }

    // No @AfterTest restore: every test that touches the registry wraps itself in withCleanBackends, which
    // saves and re-registers the incumbents. Resetting and calling registerPlatformBackends() instead only
    // worked on the JVM -- the native targets register eagerly before main, so discovery is a no-op there and
    // the platform's backend was gone for every later test.

    @Test
    fun `a registered backend is reached exactly above the level-1 threshold`() = withCleanBackends {
        val recording = Recording()
        registerBackend(recording)
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
    fun `axpy and scale route on the same threshold as dot`() = withCleanBackends {
        val recording = Recording()
        registerBackend(recording)
        val threshold = dispatchThresholds.level1
        if (threshold == Int.MAX_VALUE) return@withCleanBackends
        val v = DenseVector.of(DoubleArray(threshold) { 1.0 })
        val x = DenseVector.of(DoubleArray(threshold) { 2.0 })
        axpy(v, 3.0, x)
        scale(v, 0.5)
        assertEquals(1, recording.axpys, "axpy did not route")
        assertEquals(1, recording.scales, "scale did not route")
        assertTrue(v.data.all { it == 3.5 }, "routed arithmetic is wrong: ${v.data[0]}")
    }

    @Test
    fun `the dense reductions route but the sparse ones cannot`() = withCleanBackends {
        val recording = Recording()
        registerBackend(recording)
        val threshold = dispatchThresholds.level1
        if (threshold == Int.MAX_VALUE) return@withCleanBackends
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
     * iamax stays off the seam by design; see [VectorKernels]. Its tie-breaking and NaN ranking are koblas's own
     * contract, and `idamax` implementations disagree about the latter, so routing it would make the answer
     * depend on whether a host library happened to be installed.
     */
    @Test
    fun `iamax does not route`() = withCleanBackends {
        val recording = Recording()
        registerBackend(recording)
        val threshold = dispatchThresholds.level1
        val len = if (threshold == Int.MAX_VALUE) 4096 else threshold
        val v = DenseVector.of(DoubleArray(len) { if (it == 7) -9.0 else 1.0 })
        assertEquals(7, iamax(v))
        assertEquals(0, recording.dots + recording.nrm2s + recording.asums, "iamax reached the seam")
    }

    /**
     * Ranking and overriding, read through the context rather than the seam.
     *
     * The assertions are about names rather than identity now, because `koblas.vectorKernels` is the routed
     * pair (compiled-in plus whichever host won) rather than the registered object itself. That is the point
     * of the routing: a registered backend is *consulted* above a length, not swapped in wholesale.
     */
    @Test
    fun `registration keeps the highest priority and install overrides both`() = withCleanBackends {
        val platform = koblas.vectorKernels.name
        registerBackend(Recording(priority = 200).named("strong"))
        registerBackend(Recording(priority = 10).named("weak"))
        assertEquals("$platform+strong", koblas.vectorKernels.name, "a weaker registration displaced a stronger one")
        val override = Recording(priority = 0).named("override")
        installBackends(koblas.with(vectorKernels = override))
        assertSame(override, koblas.vectorKernels, "install must win regardless of priority, and unrouted")
        installBackends(null)
        assertEquals("$platform+strong", koblas.vectorKernels.name, "clearing the override falls back to registration")
        resetBackends()
        assertEquals(platform, koblas.vectorKernels.name, "a cleared registry leaves the compiled-in kernels")
    }

    /**
     * The compiled-in kernels satisfy the interface, which is the whole point of the change and could not
     * be asserted before: they were loose `expect fun`s, so there was no object to hand to a conformance
     * check and no way to state that they and a host backend are the same kind of thing.
     *
     * Everything runs over an offset window rather than a whole array, because the `(offset, length)`
     * contract is the part an implementation gets wrong.
     */
    @Test
    fun `the compiled-in kernels satisfy the VectorKernels contract`() {
        val k: VectorKernels = PlatformVectorKernels
        assertTrue(k.name.isNotEmpty(), "the kernels must name themselves; mathBackend reports it")

        val a = DoubleArray(40) { it * 0.5 - 3.0 }
        val b = DoubleArray(40) { 1.0 / (it + 1) }
        val off = 5
        val len = 21 // deliberately not a lane multiple, so the scalar tail runs too

        var dot = 0.0
        for (i in off until off + len) dot += a[i] * b[i]
        assertEquals(dot, k.dot(a, off, b, off, len), absoluteTolerance = 1e-12)

        var asum = 0.0
        for (i in off until off + len) asum += abs(a[i])
        assertEquals(asum, k.asum(a, off, len), absoluteTolerance = 1e-12)

        var sq = 0.0
        for (i in off until off + len) sq += a[i] * a[i]
        assertEquals(sqrt(sq), k.nrm2(a, off, len), absoluteTolerance = 1e-12)

        val y = a.copyOf()
        k.axpy(y, off, 2.0, b, off, len)
        for (i in a.indices) {
            val want = if (i in off until off + len) a[i] + 2.0 * b[i] else a[i]
            assertEquals(want, y[i], absoluteTolerance = 1e-12, message = "axpy touched outside its window at $i")
        }

        val v = a.copyOf()
        k.scale(v, off, 3.0, len)
        for (i in a.indices) {
            val want = if (i in off until off + len) a[i] * 3.0 else a[i]
            assertEquals(want, v[i], absoluteTolerance = 1e-12, message = "scale touched outside its window at $i")
        }
    }

    /**
     * `dot4` is the one member with a default, so both halves of that need pinning: the default (four [dot]
     * calls, which is what a host backend inherits) and the platform override (one pass sharing the `b`
     * loads) must agree, and both must agree with four hand-written dots.
     */
    @Test
    fun `the dot4 default and the platform override agree`() {
        val a = DoubleArray(4 * 30) { it * 0.25 - 5.0 }
        val b = DoubleArray(30) { 2.0 - it * 0.1 }
        val stride = 30
        val len = 23 // shorter than the stride, so a correct implementation reads only part of each column

        val expected = DoubleArray(4)
        for (r in 0 until 4) {
            var s = 0.0
            for (i in 0 until len) s += a[r * stride + i] * b[i]
            expected[r] = s
        }

        // The interface default. Recording implements the five required members and not dot4, so it
        // inherits it -- and being a counter, it also proves the default is four `dot` calls rather than
        // some other route. Delegating with `by PlatformVectorKernels` would NOT work here: delegation
        // forwards every member including the defaulted one, so it would silently test the override twice.
        val inherited = Recording()
        val viaDefault = DoubleArray(4)
        inherited.dot4(a, 0, stride, b, 0, len, viaDefault, 0)
        assertEquals(4, inherited.dots, "the default must reach dot once per column")

        val viaPlatform = DoubleArray(4)
        PlatformVectorKernels.dot4(a, 0, stride, b, 0, len, viaPlatform, 0)

        for (r in 0 until 4) {
            assertEquals(expected[r], viaDefault[r], absoluteTolerance = 1e-12, message = "default row $r")
            assertEquals(expected[r], viaPlatform[r], absoluteTolerance = 1e-12, message = "platform row $r")
        }
    }

    /** The rescaling contract the interface states: a plain sum of squares would overflow here. */
    @Test
    fun `the compiled-in nrm2 survives components that square out of range`() {
        val big = doubleArrayOf(3e200, 4e200)
        assertEquals(5e200, PlatformVectorKernels.nrm2(big, 0, 2), absoluteTolerance = 1e188)
        val tiny = doubleArrayOf(3e-200, 4e-200)
        assertEquals(5e-200, PlatformVectorKernels.nrm2(tiny, 0, 2), absoluteTolerance = 1e-212)
    }

    /** With nothing registered the routed kernels are the platform ones, and mathBackend says so. */
    @Test
    fun `the routed kernels report the platform name and gain a suffix for a host`() = withCleanBackends {
        assertEquals(PlatformVectorKernels.name, koblas.vectorKernels.name)
        assertEquals(koblas.vectorKernels.name, mathBackend, "mathBackend is the routed kernels' name")
        registerBackend(Recording(priority = 90))
        assertEquals("${PlatformVectorKernels.name}+recording", koblas.vectorKernels.name)
        resetBackends()
        assertEquals(PlatformVectorKernels.name, koblas.vectorKernels.name)
    }

    /** The reset hook clears the override too, not only the registration. */
    @Test
    fun `the reset hook clears the install override too`() = withCleanBackends {
        val platform = koblas.vectorKernels.name
        installBackends(koblas.with(vectorKernels = Recording(priority = 0).named("override")))
        resetBackends()
        assertEquals(platform, koblas.vectorKernels.name, "reset must clear the override, not just registration")
    }
}
