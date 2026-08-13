package com.eignex.koblas.dense

import com.eignex.koblas.DenseVector
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

class VectorKernelsTest {

    private class Recording(override val priority: Int = 90) : VectorKernels {
        override var name: String = "recording"
            private set

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
        val before = recording.dots
        val short1 = DenseVector.of(doubleArrayOf(3.0))
        val short2 = DenseVector.of(doubleArrayOf(4.0))
        assertEquals(12.0, short1 dot short2)
        assertEquals(before, recording.dots, "a length-1 dot should stay portable, but it reached the backend")
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
        val sparse = SparseVector(threshold, IntArray(threshold) { it }, DoubleArray(threshold) { 3.0 })
        norm2(sparse)
        asum(sparse)
        assertEquals(1, recording.nrm2s, "norm2 routed a sparse vector")
        assertEquals(1, recording.asums, "asum routed a sparse vector")
    }

    /** iamax stays off the seam, since `idamax` implementations disagree about NaN and koblas pins its own contract. */
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

        // Recording omits dot4, so this measures the interface default. Delegating with `by PlatformVectorKernels`
        // would forward the defaulted member and silently test the override twice.
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

    @Test
    fun `the compiled-in nrm2 survives components that square out of range`() {
        val big = doubleArrayOf(3e200, 4e200)
        assertEquals(5e200, PlatformVectorKernels.nrm2(big, 0, 2), absoluteTolerance = 1e188)
        val tiny = doubleArrayOf(3e-200, 4e-200)
        assertEquals(5e-200, PlatformVectorKernels.nrm2(tiny, 0, 2), absoluteTolerance = 1e-212)
    }

    @Test
    fun `the routed kernels report the platform name and gain a suffix for a host`() = withCleanBackends {
        assertEquals(PlatformVectorKernels.name, koblas.vectorKernels.name)
        assertEquals(koblas.vectorKernels.name, mathBackend, "mathBackend is the routed kernels' name")
        registerBackend(Recording(priority = 90))
        assertEquals("${PlatformVectorKernels.name}+recording", koblas.vectorKernels.name)
        resetBackends()
        assertEquals(PlatformVectorKernels.name, koblas.vectorKernels.name)
    }
}
