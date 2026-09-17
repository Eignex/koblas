package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Which kernel the Vector API arm reaches, which is not the same answer for every operation.
 *
 * Only the reductions have a vector kernel here. The elementwise operations are ordinary Kotlin loops that
 * HotSpot vectorises on its own, and hand-written lanes measured no faster and in one case half the speed, so
 * they were removed. A caller cannot see that from a result, since both kernels compute the same thing, which
 * is why what the arm reports about itself is worth a test: this is the difference between an arm that dropped
 * work it did not need and one that quietly stopped vectorising.
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
    fun `elementwise operations stay with the portable loops at every width`() {
        if (!SimdVectorKernels.isAvailable) return skipped()

        for (operation in listOf(
            DenseOperation.Axpy,
            DenseOperation.Scale,
            DenseOperation.Swap,
            DenseOperation.Rot,
        )) {
            assertEquals(ScalarVectorKernels.name, reached(operation, WIDE), "$operation named a vector kernel")
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
