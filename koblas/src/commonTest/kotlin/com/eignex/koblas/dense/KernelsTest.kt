package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

class KernelsTest {

    private class Recording(override val priority: Int = 90) : F64Kernels {
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
        var swaps = 0

        override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
            swaps++
            for (i in 0 until len) {
                val t = a[aOff + i]
                a[aOff + i] = b[bOff + i]
                b[bOff + i] = t
            }
        }

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
    fun `a registered backend handles vector runs at and above the crossover`() = withCleanBackends {
        val recording = Recording()
        registerBackend(recording)
        val x = F64DenseVector.of(DoubleArray(16) { 1.0 })
        val y = F64DenseVector.of(DoubleArray(16) { 2.0 })
        assertEquals(32.0, x dot y)
        assertEquals(0, recording.dots, "a tiny dot must stay on compiled-in kernels")
        val short1 = F64DenseVector.of(doubleArrayOf(3.0))
        val short2 = F64DenseVector.of(doubleArrayOf(4.0))
        assertEquals(12.0, short1 dot short2)
        assertEquals(0, recording.dots, "a length-1 dot must stay on compiled-in kernels")
        val boundary = F64DenseVector.of(DoubleArray(64) { 1.0 })
        assertEquals(64.0, boundary dot boundary)
        assertEquals(1, recording.dots, "a length-64 dot did not reach the selected backend")
    }

    @Test
    fun `axpy and scale route to the selected backend at the crossover`() = withCleanBackends {
        val recording = Recording()
        registerBackend(recording)
        val v = F64DenseVector.of(DoubleArray(64) { 1.0 })
        val x = F64DenseVector.of(DoubleArray(64) { 2.0 })
        v.axpy(3.0, x)
        v.scale(0.5)
        assertEquals(1, recording.axpys, "axpy did not route")
        assertEquals(1, recording.scales, "scale did not route")
        assertTrue(v.data.all { it == 3.5 }, "routed arithmetic is wrong: ${v.data[0]}")
    }

    @Test
    fun `the platform arithmetic axpy does not take the DAXPY zero return`() {
        val x = DoubleArray(64).also { it[17] = Double.POSITIVE_INFINITY }
        val y = DoubleArray(64)

        axpyArithmetic(F64PlatformKernels, y, 0, 0.0, x, 0, x.size)

        assertTrue(y[17].isNaN())
    }

    @Test
    fun `the dense reductions route but the sparse ones cannot`() = withCleanBackends {
        val recording = Recording()
        registerBackend(recording)
        val dense = F64DenseVector.of(DoubleArray(64) { 3.0 })
        dense.norm2()
        dense.asum()
        assertEquals(1, recording.nrm2s, "norm2 on a dense vector did not route")
        assertEquals(1, recording.asums, "asum on a dense vector did not route")
        val sparse = F64SparseVector(8, IntArray(8) { it }, DoubleArray(8) { 3.0 })
        sparse.norm2()
        sparse.asum()
        assertEquals(1, recording.nrm2s, "norm2 routed a sparse vector")
        assertEquals(1, recording.asums, "asum routed a sparse vector")
    }

    /** iamax stays off the seam, since `idamax` implementations disagree about NaN and koblas pins its own contract. */
    @Test
    fun `iamax does not route`() = withCleanBackends {
        val recording = Recording()
        registerBackend(recording)
        val len = 16
        val v = F64DenseVector.of(DoubleArray(len) { if (it == 7) -9.0 else 1.0 })
        assertEquals(7, v.iamax())
        assertEquals(0, recording.dots + recording.nrm2s + recording.asums, "iamax reached the seam")
    }

    @Test
    fun `registration keeps the highest priority and install overrides both`() = withCleanBackends {
        val platform = koblas.kernels.name
        registerBackend(Recording(priority = 200).named("strong"))
        registerBackend(Recording(priority = 10).named("weak"))
        assertEquals("$platform+strong", koblas.kernels.name, "a weaker registration displaced a stronger one")
        val override = Recording(priority = 0).named("override")
        installBackends(koblas.with(kernels = override))
        assertSame(override, koblas.kernels, "install must win regardless of priority, and unrouted")
        installBackends(null)
        assertEquals("$platform+strong", koblas.kernels.name, "clearing the override falls back to registration")
        resetBackends()
        assertEquals(platform, koblas.kernels.name, "a cleared registry leaves the compiled-in kernels")
    }

    @Test
    fun `the compiled-in kernels satisfy the F64Kernels contract`() {
        val k: F64Kernels = F64PlatformKernels
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
     * The router carries one override per seam member it routes, and a member it forgets falls through to
     * the interface default rather than reaching anything.
     */
    @Test
    fun `every routed kernel reaches the host it was given`() {
        val recording = Recording()
        val routed: F64Kernels = F64RoutedKernels(recording)
        val a = DoubleArray(64) { it.toDouble() }
        val b = DoubleArray(64) { 1.0 }
        routed.dot(a, 0, b, 0, 64)
        routed.axpy(a, 0, 2.0, b, 0, 64)
        routed.scale(a, 0, 2.0, 64)
        routed.nrm2(a, 0, 64)
        routed.asum(a, 0, 64)
        routed.swap(a, 0, b, 0, 64)
        assertEquals(1, recording.dots, "dot did not reach the host")
        assertEquals(1, recording.axpys, "axpy did not reach the host")
        assertEquals(1, recording.scales, "scale did not reach the host")
        assertEquals(1, recording.nrm2s, "nrm2 did not reach the host")
        assertEquals(1, recording.asums, "asum did not reach the host")
        assertEquals(1, recording.swaps, "swap did not reach the host, so the router is missing an override")
    }

    @Test
    fun `routed kernels respect the crossover boundary`() {
        val recording = Recording()
        val routed: F64Kernels = F64RoutedKernels(recording)
        val a = DoubleArray(65) { it.toDouble() }
        val b = DoubleArray(65) { 1.0 }

        routed.dot(a, 0, b, 0, 63)
        routed.axpy(a, 0, 2.0, b, 0, 63)
        routed.scale(a, 0, 2.0, 63)
        routed.nrm2(a, 0, 63)
        routed.asum(a, 0, 63)
        routed.swap(a, 0, b, 0, 63)
        assertEquals(
            0,
            recording.dots + recording.axpys + recording.scales + recording.nrm2s + recording.asums + recording.swaps,
        )

        routed.dot(a, 0, b, 0, 64)
        routed.axpy(a, 0, 2.0, b, 0, 64)
        routed.scale(a, 0, 2.0, 64)
        routed.nrm2(a, 0, 64)
        routed.asum(a, 0, 64)
        routed.swap(a, 0, b, 0, 64)
        assertEquals(1, recording.dots, "dot at the crossover did not reach the host")
        assertEquals(1, recording.axpys, "axpy at the crossover did not reach the host")
        assertEquals(1, recording.scales, "scale at the crossover did not reach the host")
        assertEquals(1, recording.nrm2s, "nrm2 at the crossover did not reach the host")
        assertEquals(1, recording.asums, "asum at the crossover did not reach the host")
        assertEquals(1, recording.swaps, "swap at the crossover did not reach the host")

        routed.dot(a, 0, b, 0, 65)
        routed.axpy(a, 0, 2.0, b, 0, 65)
        routed.scale(a, 0, 2.0, 65)
        routed.nrm2(a, 0, 65)
        routed.asum(a, 0, 65)
        routed.swap(a, 0, b, 0, 65)
        assertEquals(2, recording.dots, "dot above the crossover did not reach the host")
        assertEquals(2, recording.axpys, "axpy above the crossover did not reach the host")
        assertEquals(2, recording.scales, "scale above the crossover did not reach the host")
        assertEquals(2, recording.nrm2s, "nrm2 above the crossover did not reach the host")
        assertEquals(2, recording.asums, "asum above the crossover did not reach the host")
        assertEquals(2, recording.swaps, "swap above the crossover did not reach the host")
    }

    @Test
    fun `routed kernels keep public scale and axpy noops`() {
        val recording = Recording()
        val routed: F64Kernels = F64RoutedKernels(recording)
        val x = DoubleArray(64) { Double.POSITIVE_INFINITY }
        val y = DoubleArray(64)

        routed.axpy(y, 0, 0.0, x, 0, 64)
        routed.scale(x, 0, 1.0, 64)

        assertEquals(0, recording.axpys, "zero axpy reached the host")
        assertEquals(0, recording.scales, "unit scale reached the host")
        assertTrue(y.all { it == 0.0 }, "zero axpy must not evaluate infinity times zero")
        assertTrue(x.all { it == Double.POSITIVE_INFINITY }, "unit scale changed the vector")
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

        // Recording omits dot4, so this measures the interface default. Delegating with `by F64PlatformKernels`
        // would forward the defaulted member and silently test the override twice.
        val inherited = Recording()
        val viaDefault = DoubleArray(4)
        inherited.dot4(a, 0, stride, b, 0, len, viaDefault, 0)
        assertEquals(4, inherited.dots, "the default must reach dot once per column")

        val viaPlatform = DoubleArray(4)
        F64PlatformKernels.dot4(a, 0, stride, b, 0, len, viaPlatform, 0)

        for (r in 0 until 4) {
            assertEquals(expected[r], viaDefault[r], absoluteTolerance = 1e-12, message = "default row $r")
            assertEquals(expected[r], viaPlatform[r], absoluteTolerance = 1e-12, message = "platform row $r")
        }
    }

    @Test
    fun `the compiled-in nrm2 survives components that square out of range`() {
        val big = doubleArrayOf(3e200, 4e200)
        assertEquals(5e200, F64PlatformKernels.nrm2(big, 0, 2), absoluteTolerance = 1e188)
        val tiny = doubleArrayOf(3e-200, 4e-200)
        assertEquals(5e-200, F64PlatformKernels.nrm2(tiny, 0, 2), absoluteTolerance = 1e-212)
    }

    @Test
    fun `nrm2 survives out-of-range components at lengths that vectorize`() {
        for (len in intArrayOf(16, 33, 64)) {
            val big = DoubleArray(len) { 1e200 }
            val expected = sqrt(len.toDouble()) * 1e200
            assertEquals(expected, F64PlatformKernels.nrm2(big, 0, len), absoluteTolerance = expected * 1e-12)
            val tiny = DoubleArray(len) { 1e-200 }
            val expectedTiny = sqrt(len.toDouble()) * 1e-200
            assertEquals(
                expectedTiny,
                F64PlatformKernels.nrm2(tiny, 0, len),
                absoluteTolerance = expectedTiny * 1e-12,
            )
        }
    }

    @Test
    fun `nrm2 agrees with the portable norm across offsets and lengths`() {
        val rng = Random(20260815)
        val v = DoubleArray(300) { rng.nextDouble(-1.0, 1.0) }
        for (off in intArrayOf(0, 1, 7)) {
            for (len in intArrayOf(0, 1, 3, 8, 31, 128, 293)) {
                val expected = euclideanNorm(v, off, len)
                assertEquals(
                    expected,
                    F64PlatformKernels.nrm2(v, off, len),
                    absoluteTolerance = 1e-12 * (expected + 1.0),
                    message = "off $off len $len",
                )
            }
        }
    }

    @Test
    fun `the routed kernels report the platform name and gain a suffix for a host`() = withCleanBackends {
        assertEquals(F64PlatformKernels.name, koblas.kernels.name)
        assertEquals(koblas.kernels.name, mathBackend, "mathBackend is the routed kernels' name")
        registerBackend(Recording(priority = 90))
        assertEquals("${F64PlatformKernels.name}+recording", koblas.kernels.name)
        resetBackends()
        assertEquals(F64PlatformKernels.name, koblas.kernels.name)
    }

    /**
     * The triangular and Householder kernels reach their last row with an empty tail, so every routine here
     * is called with a zero length. Each must read nothing and return the identity.
     */
    @Test
    fun `every kernel accepts a zero length run`() {
        val kernels = platformKernels
        val v = doubleArrayOf(1.0, 2.0, 3.0)
        assertEquals(0.0, kernels.dot(v, 3, v, 3, 0), "dot over nothing")
        assertEquals(0.0, kernels.nrm2(v, 3, 0), "nrm2 over nothing")
        assertEquals(0.0, kernels.asum(v, 3, 0), "asum over nothing")
        kernels.axpy(v, 3, 2.0, v, 3, 0)
        kernels.scale(v, 3, 2.0, 0)
        assertEquals(listOf(1.0, 2.0, 3.0), v.toList(), "a zero-length write touched the vector")
        val quads = DoubleArray(4)
        kernels.dot4(v, 3, 0, v, 3, 0, quads, 0)
        assertEquals(listOf(0.0, 0.0, 0.0, 0.0), quads.toList(), "dot4 over nothing")
    }

    @Test
    fun `the compiled-in level-1 kernels agree with the scalar loops`() =
        assertLevel1KernelsAgreeWithScalar(F64PlatformKernels)

    @Test
    fun `the compiled-in reductions agree with the scalar loops`() = assertReductionsAgreeWithScalar(F64PlatformKernels)

    @Test
    fun `the compiled-in swap agrees with the scalar loop`() = assertSwapAgreesWithScalar(F64PlatformKernels)
}
