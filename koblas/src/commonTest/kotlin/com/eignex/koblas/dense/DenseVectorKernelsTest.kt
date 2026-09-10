package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

class DenseVectorKernelsTest {

    @Test
    fun `the scalar and platform arithmetic axpy kernels agree`() {
        val source = DoubleArray(47) { it * 0.125 - 2.0 }
        val initial = DoubleArray(53) { 3.0 - it * 0.2 }
        val expected = initial.copyOf()
        val actual = initial.copyOf()

        ScalarPanelKernels.axpyArithmetic(expected, 7, -0.75, source, 3, 31)
        platformDenseKernelFamilies.panel.axpyArithmetic(actual, 7, -0.75, source, 3, 31)

        assertClose(expected, actual, "arithmetic axpy")
    }

    @Test
    fun `the platform arithmetic axpy does not take the DAXPY zero return`() {
        val x = DoubleArray(64).also { it[17] = Double.POSITIVE_INFINITY }
        val y = DoubleArray(64)

        axpyArithmetic(platformDenseKernelFamilies.panel, y, 0, 0.0, x, 0, x.size)

        assertTrue(y[17].isNaN())
    }

    @Test
    fun `the compiled-in kernels satisfy the dense vector contract`() {
        val k: DenseVectorKernels = PlatformVectorKernels
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
    fun `the compiled-in kernels keep the public scale and axpy noops`() {
        val x = DoubleArray(64) { Double.POSITIVE_INFINITY }
        val y = DoubleArray(64)

        PlatformVectorKernels.axpy(y, 0, 0.0, x, 0, 64)
        PlatformVectorKernels.scale(x, 0, 1.0, 64)

        assertTrue(y.all { it == 0.0 }, "zero axpy must not evaluate infinity times zero")
        assertTrue(x.all { it == Double.POSITIVE_INFINITY }, "unit scale changed the vector")
    }

    @Test
    fun `the scalar and platform dot4 kernels agree`() {
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

        val viaScalar = DoubleArray(4)
        ScalarPanelKernels.dot4(a, 0, stride, b, 0, len, viaScalar, 0)

        val viaPlatform = DoubleArray(4)
        platformDenseKernelFamilies.panel.dot4(a, 0, stride, b, 0, len, viaPlatform, 0)

        for (r in 0 until 4) {
            assertEquals(expected[r], viaScalar[r], absoluteTolerance = 1e-12, message = "scalar row $r")
            assertEquals(expected[r], viaPlatform[r], absoluteTolerance = 1e-12, message = "platform row $r")
        }
    }

    @Test
    fun `the scalar and platform axpy4 kernels agree`() {
        val stride = 31
        val a = DoubleArray(4 * stride) { it * 0.125 - 4.0 }
        a[7] = Double.POSITIVE_INFINITY
        val coefficients = doubleArrayOf(0.0, -2.0, 0.5, 3.0)
        val initial = DoubleArray(35) { it * 0.25 }
        val expected = initial.copyOf()
        val off = 3
        val len = 23
        for (i in 0 until len) {
            for (r in 0 until 4) expected[off + i] += coefficients[r] * a[r * stride + i]
        }

        val viaScalar = initial.copyOf()
        ScalarPanelKernels.axpy4(
            viaScalar, off, a, 0, stride,
            coefficients[0], coefficients[1], coefficients[2], coefficients[3], len,
        )
        val viaPlatform = initial.copyOf()
        platformDenseKernelFamilies.panel.axpy4(
            viaPlatform, off, a, 0, stride,
            coefficients[0], coefficients[1], coefficients[2], coefficients[3], len,
        )

        for (i in expected.indices) {
            if (expected[i].isNaN()) {
                assertTrue(viaScalar[i].isNaN(), "scalar index $i")
                assertTrue(viaPlatform[i].isNaN(), "platform index $i")
            } else {
                assertEquals(expected[i], viaScalar[i], absoluteTolerance = 1e-12, message = "scalar index $i")
                assertEquals(expected[i], viaPlatform[i], absoluteTolerance = 1e-12, message = "platform index $i")
            }
        }
    }

    @Test
    fun `the scalar and platform dotAxpy kernels agree with aliased runs`() {
        val a = DoubleArray(37) { it * 0.2 - 2.5 }
        val initial = DoubleArray(39) { 3.0 - it * 0.1 }
        val off = 5
        val len = 27
        val alpha = -0.75
        var expectedDot = 0.0
        val expected = initial.copyOf()
        for (i in 0 until len) {
            val ai = a[2 + i]
            val xi = initial[off + i]
            expectedDot += ai * xi
            expected[off + i] += alpha * ai
        }

        val viaScalar = initial.copyOf()
        val scalarDot = ScalarPanelKernels.dotAxpy(viaScalar, off, alpha, a, 2, viaScalar, off, len)
        val viaPlatform = initial.copyOf()
        val platformDot = platformDenseKernelFamilies.panel.dotAxpy(
            viaPlatform,
            off,
            alpha,
            a,
            2,
            viaPlatform,
            off,
            len,
        )

        assertEquals(expectedDot, scalarDot, absoluteTolerance = 1e-12)
        assertEquals(expectedDot, platformDot, absoluteTolerance = 1e-12)
        assertContentEquals(expected, viaScalar)
        for (i in expected.indices) {
            assertEquals(expected[i], viaPlatform[i], absoluteTolerance = 1e-12, message = "platform index $i")
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
    fun `nrm2 survives out-of-range components at lengths that vectorize`() {
        for (len in intArrayOf(16, 33, 64)) {
            val big = DoubleArray(len) { 1e200 }
            val expected = sqrt(len.toDouble()) * 1e200
            assertEquals(expected, PlatformVectorKernels.nrm2(big, 0, len), absoluteTolerance = expected * 1e-12)
            val tiny = DoubleArray(len) { 1e-200 }
            val expectedTiny = sqrt(len.toDouble()) * 1e-200
            assertEquals(
                expectedTiny,
                PlatformVectorKernels.nrm2(tiny, 0, len),
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
                    PlatformVectorKernels.nrm2(v, off, len),
                    absoluteTolerance = 1e-12 * (expected + 1.0),
                    message = "off $off len $len",
                )
            }
        }
    }

    @Test
    fun `the context reports the selected kernels by name`() {
        assertEquals(PlatformVectorKernels.name, koblas.vectorKernels.name)
        assertEquals(koblas.vectorKernels.name, mathBackend, "mathBackend is the selected kernels' name")
    }

    /**
     * The triangular and Householder kernels reach their last row with an empty tail, so every routine here
     * is called with a zero length. Each must read nothing and return the identity.
     */
    @Test
    fun `every kernel accepts a zero length run`() {
        val vectorKernels = PlatformVectorKernels
        val panelKernels = platformDenseKernelFamilies.panel
        val v = doubleArrayOf(1.0, 2.0, 3.0)
        assertEquals(0.0, vectorKernels.dot(v, 3, v, 3, 0), "dot over nothing")
        assertEquals(0.0, vectorKernels.nrm2(v, 3, 0), "nrm2 over nothing")
        assertEquals(0.0, vectorKernels.asum(v, 3, 0), "asum over nothing")
        vectorKernels.axpy(v, 3, 2.0, v, 3, 0)
        vectorKernels.scale(v, 3, 2.0, 0)
        assertEquals(listOf(1.0, 2.0, 3.0), v.toList(), "a zero-length write touched the vector")
        val quads = DoubleArray(4)
        panelKernels.dot4(v, 3, 0, v, 3, 0, quads, 0)
        assertEquals(listOf(0.0, 0.0, 0.0, 0.0), quads.toList(), "dot4 over nothing")
        panelKernels.axpy4(v, 3, v, 3, 0, 1.0, 2.0, 3.0, 4.0, 0)
        assertEquals(0.0, panelKernels.dotAxpy(v, 3, 2.0, v, 3, v, 3, 0), "dotAxpy over nothing")
        assertEquals(listOf(1.0, 2.0, 3.0), v.toList(), "a fused zero-length write touched the vector")
    }

    @Test
    fun `the compiled-in level-1 kernels agree with the scalar loops`() =
        assertLevel1KernelsAgreeWithReference(PlatformVectorKernels)

    @Test
    fun `ssqd stays exact where the expanded form cancels`() {
        // The identity sum(a - b)^2 == a.a - 2a.b + b.b holds in exact arithmetic and not in doubles: at
        // this magnitude the three dots agree to fewer digits than the answer has, so an implementation
        // that took the shortcut cannot return 1.0 here.
        val a = doubleArrayOf(1e8, 1e8, 1e8)
        val b = doubleArrayOf(1e8 + 1.0, 1e8, 1e8)
        val expanded = PlatformVectorKernels.dot(a, 0, a, 0, 3) -
            2.0 * PlatformVectorKernels.dot(a, 0, b, 0, 3) +
            PlatformVectorKernels.dot(b, 0, b, 0, 3)
        assertEquals(1.0, PlatformVectorKernels.ssqd(a, 0, b, 0, 3), "fused")
        assertTrue(abs(expanded - 1.0) > 1e-3, "the expanded form should be the inexact one here: $expanded")
    }

    @Test
    fun `ssqd is symmetric and zero on equal or empty runs`() {
        val rng = Random(20260903)
        for (len in intArrayOf(0, 1, 5, 64, 130)) {
            val a = DoubleArray(len) { rng.nextDouble(-1.0, 1.0) }
            val b = DoubleArray(len) { rng.nextDouble(-1.0, 1.0) }
            assertEquals(
                PlatformVectorKernels.ssqd(a, 0, b, 0, len),
                PlatformVectorKernels.ssqd(b, 0, a, 0, len),
                "symmetry len=$len",
            )
            assertEquals(0.0, PlatformVectorKernels.ssqd(a, 0, a, 0, len), "equal runs len=$len")
        }
    }

    @Test
    fun `the compiled-in reductions agree with the scalar loops`() =
        assertReductionsAgreeWithReference(PlatformVectorKernels)

    @Test
    fun `the compiled-in swap agrees with the scalar loop`() = assertSwapAgreesWithReference(PlatformVectorKernels)

    @Test
    fun `the compiled-in modified Givens kernels agree with the portable ones`() {
        assertModifiedGivensKernelsAgreeWithReference(PlatformVectorKernels)
        assertRotKernelAgreesWithReference(PlatformVectorKernels)
    }
}
