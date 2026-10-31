package com.eignex.koblas.dense

import com.eignex.koblas.internal.dense.ScalarVectorKernels
import com.eignex.koblas.internal.jvm25.simd.SimdOps
import com.eignex.koblas.internal.jvm25.simd.SimdVectorKernels
import com.eignex.koblas.internal.sparse.axpy
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Checks per-operation routing on the Vector API arm. Reductions and AXPY use explicit vector kernels;
 * other elementwise operations use Kotlin loops that HotSpot vectorises. Routes distinguish these bodies
 * even when their numerical results agree.
 */
class SimdVectorKernelsTest {
    private val kernels = SimdVectorKernels

    private fun reached(operation: DenseOperation, length: Int): String? =
        kernels.implementationFor(operation, length, contiguous = true)

    @Test
    fun `reductions reach the vector kernel once there are lanes to fill`() {
        if (!SimdVectorKernels.isAvailable) return skipped()

        for (operation in listOf(DenseOperation.Dot, DenseOperation.Sum, DenseOperation.Asum)) {
            assertEquals(kernels.name, reached(operation, WIDE), "$operation stayed scalar at $WIDE")
        }
    }

    @Test
    fun `contiguous elementwise operations have explicit vector paths`() {
        if (!SimdVectorKernels.isAvailable) return skipped()

        for (operation in listOf(
            DenseOperation.Scale,
            DenseOperation.Swap,
            DenseOperation.Rot,
        )) {
            assertEquals(SimdVectorKernels.name, reached(operation, WIDE), "$operation stayed scalar")
        }
    }

    @Test
    fun `contiguous axpy has an explicit vector path`() {
        if (!SimdVectorKernels.isAvailable) return skipped()

        assertEquals(kernels.name, reached(DenseOperation.Axpy, WIDE))
        assertLevel1KernelsAgreeWithReference(kernels)
    }

    private fun assertAxpyAgreesWithReference(expected: DoubleArray, actual: DoubleArray) =
        assertContentEquals(expected, actual)

    @Test
    fun `axpy short tails agree with the scalar oracle on shared buffers`() {
        if (!SimdVectorKernels.isAvailable) return skipped()
        val lane = SimdOps.lanes()
        val values = doubleArrayOf(
            -0.125,
            0.0,
            0.25,
            -1.25,
            -2.5,
            3.0,
        )
        for (length in (0..2 * lane + 1).toList() + listOf(27, 31, 56, 64, 65, 70)) {
            for (alpha in listOf(0.0, 1.0, -0.75, 2.0)) {
                for (sourceOffset in listOf(3, length + 6)) {
                    val expected = DoubleArray(2 * length + 9) { values[it % values.size] }
                    val actual = expected.copyOf()
                    ScalarVectorKernels.axpy(expected, 3, alpha, expected, sourceOffset, length)

                    kernels.axpy(actual, 3, alpha, actual, sourceOffset, length)

                    assertAxpyAgreesWithReference(expected, actual)
                }
            }
        }
    }

    @Test
    fun `unit axpy agrees with the scalar oracle across windows and shared buffers`() {
        if (!SimdVectorKernels.isAvailable) return skipped()
        val values = doubleArrayOf(
            -0.125,
            0.0,
            0.25,
            -1.25,
            -2.5,
            3.0,
        )
        for (length in listOf(2, 6, 8, 27, 56, 64, 70, 65_535, 65_536)) {
            for (shared in listOf(false, true)) {
                val expected = DoubleArray(2 * length + 9) { values[it % values.size] }
                val actual = expected.copyOf()
                val source = DoubleArray(expected.size) { values[(it + 3) % values.size] }
                for (sourceOffset in listOf(3, length + 6)) {
                    ScalarVectorKernels.axpy(expected, 3, 1.0, if (shared) expected else source, sourceOffset, length)

                    kernels.axpy(actual, 3, 1.0, if (shared) actual else source, sourceOffset, length)

                    assertAxpyAgreesWithReference(expected, actual)
                }
            }
        }
    }

    @Test
    fun `the index search waits for its own measured width`() {
        if (!SimdVectorKernels.isAvailable) return skipped()

        // Below its crossover the vectorised search is inside the noise and loses at one width; see the
        // constant's own documentation for the measurement.
        assertEquals(ScalarVectorKernels.name, reached(DenseOperation.Iamax, 128))
        assertEquals(kernels.name, reached(DenseOperation.Iamax, WIDE))
    }

    @Test
    fun `a strided run is scalar work whatever the operation`() {
        if (!SimdVectorKernels.isAvailable) return skipped()

        for (operation in DenseOperation.entries) {
            assertEquals(
                ScalarVectorKernels.name,
                kernels.implementationFor(operation, WIDE, contiguous = false),
                "$operation named a vector kernel for a strided run",
            )
        }
    }

    @Test
    fun `the norm answers with its values rather than its width`() {
        if (!SimdVectorKernels.isAvailable) return skipped()

        // The squared sum is tried first and abandoned when it leaves the normal range, so no kernel can be
        // named from the width alone.
        assertTrue(reached(DenseOperation.Nrm2, WIDE) == null)
    }

    private fun skipped() {
        println("SKIPPED: the Vector API module did not resolve, so there is no arm to ask")
    }

    private companion object {
        /** Wider than any crossover here, so a kernel that exists is reached. */
        const val WIDE = 1024
    }
}
