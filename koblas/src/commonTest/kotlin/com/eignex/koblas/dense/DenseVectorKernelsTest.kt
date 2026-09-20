package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

class DenseVectorKernelsTest {
    @Test
    fun `the platform scaling preserves offsets tails and exceptional values`() =
        assertScaleHonoursItsContract(koblas.vectorKernels)

    /** The vectorised kernels fall back to the portable ones, so they owe exact agreement, not just the contract. */
    @Test
    fun `the simd scaling agrees exactly with the kernels it falls back to`() {
        val simd = BuiltinEngines.simd ?: return
        assertScaleAgreesWithReference(simd.vectorKernels)
    }

    @Test
    fun `the compiled-in kernels name themselves for engine attribution`() {
        assertTrue(koblas.vectorKernels.name.isNotEmpty())
    }

    @Test
    fun `the compiled-in kernels keep the public scale and axpy noops`() {
        val x = DoubleArray(64) { Double.POSITIVE_INFINITY }
        val y = DoubleArray(64)

        koblas.vectorKernels.axpy(y, 0, 0.0, x, 0, 64)
        koblas.vectorKernels.scale(x, 0, 1.0, 64)

        assertTrue(y.all { it == 0.0 }, "zero axpy must not evaluate infinity times zero")
        assertTrue(x.all { it == Double.POSITIVE_INFINITY }, "unit scale changed the vector")
    }

    @Test
    fun `nrm2 survives components that square out of range at every length`() {
        val pair = doubleArrayOf(3e200, 4e200)
        assertEquals(5e200, koblas.vectorKernels.nrm2(pair, 0, 2), absoluteTolerance = 1e188)
        val tinyPair = doubleArrayOf(3e-200, 4e-200)
        assertEquals(5e-200, koblas.vectorKernels.nrm2(tinyPair, 0, 2), absoluteTolerance = 1e-212)

        for (len in intArrayOf(16, 33, 64)) {
            val big = DoubleArray(len) { 1e200 }
            val expected = sqrt(len.toDouble()) * 1e200
            assertEquals(
                expected,
                koblas.vectorKernels.nrm2(big, 0, len),
                absoluteTolerance = expected * 1e-12,
            )
            val tiny = DoubleArray(len) { 1e-200 }
            val expectedTiny = sqrt(len.toDouble()) * 1e-200
            assertEquals(
                expectedTiny,
                koblas.vectorKernels.nrm2(tiny, 0, len),
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
                val expected = euclideanNorm(v, off, 1, len)
                assertEquals(
                    expected,
                    koblas.vectorKernels.nrm2(v, off, len),
                    absoluteTolerance = 1e-12 * (expected + 1.0),
                    message = "off $off len $len",
                )
            }
        }
    }

    /**
     * A caller walking a matrix reaches its last row or column with an empty tail, so every routine here is
     * called with a zero length. Each must read nothing and return the identity.
     */
    @Test
    fun `every kernel accepts a zero length run`() {
        val vectorKernels = koblas.vectorKernels
        val v = doubleArrayOf(1.0, 2.0, 3.0)
        assertEquals(0.0, vectorKernels.dot(v, 3, v, 3, 0), "dot over nothing")
        assertEquals(0.0, vectorKernels.nrm2(v, 3, 0), "nrm2 over nothing")
        assertEquals(0.0, vectorKernels.asum(v, 3, 0), "asum over nothing")
        assertEquals(0.0, vectorKernels.sum(v, 3, 0), "sum over nothing")
        assertEquals(-1, vectorKernels.iamax(v, 3, 0), "iamax over nothing")
        vectorKernels.axpy(v, 3, 2.0, v, 3, 0)
        vectorKernels.scale(v, 3, 2.0, 0)
        vectorKernels.swap(v, 3, v, 3, 0)
        assertEquals(listOf(1.0, 2.0, 3.0), v.toList(), "a zero-length write touched the vector")
    }

    @Test
    fun `the compiled-in level-1 kernels agree with the scalar loops`() =
        assertLevel1KernelsAgreeWithReference(koblas.vectorKernels)

    @Test
    fun `the compiled in iamax honours its contract`() = assertIamaxHonoursItsContract(koblas.vectorKernels)

    /** The vectorised kernels fall back to the portable ones, so they owe exact agreement, not just the contract. */
    @Test
    fun `the simd iamax agrees exactly with the kernels it falls back to`() {
        val simd = BuiltinEngines.simd ?: return
        assertIamaxAgreesWithReference(simd.vectorKernels)
    }

    @Test
    fun `the compiled-in reductions agree with the scalar loops`() =
        assertReductionsAgreeWithReference(koblas.vectorKernels)

    @Test
    fun `the compiled-in swap agrees with the scalar loop`() = assertSwapAgreesWithReference(koblas.vectorKernels)

    @Test
    fun `the compiled-in rot kernel agrees with the portable one`() =
        assertRotKernelAgreesWithReference(koblas.vectorKernels)
}
