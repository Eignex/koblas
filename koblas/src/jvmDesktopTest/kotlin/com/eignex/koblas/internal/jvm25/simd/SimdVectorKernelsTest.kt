package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.engine.implementationFor
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

private val selectedBackend = KoblasEngine.simd ?: ScalarEngine

/**
 * Checks per-operation routing on the Vector API arm. Reductions and AXPY use explicit vector kernels;
 * other elementwise operations use Kotlin loops that HotSpot vectorises. Routes distinguish these bodies
 * even when their numerical results agree.
 */
class SimdVectorKernelsTest {
    private val kernels = selectedBackend

    private fun reached(operation: EngineImplementation, length: Int): String? =
        kernels.implementationFor(operation, length, contiguous = true)

    @Test
    fun `reductions reach the vector kernel once there are lanes to fill`() {
        if (!simdAvailable) return skipped()

        for (operation in listOf(EngineImplementation.Dot, EngineImplementation.Sum, EngineImplementation.Asum)) {
            assertEquals(
                simdKernels.vectors.name,
                reached(operation, WIDE),
                "$operation stayed scalar at $WIDE",
            )
        }
    }

    @Test
    fun `elementwise operations without vector bodies stay with the portable loops`() {
        if (!simdAvailable) return skipped()

        for (operation in listOf(
            EngineImplementation.Scale,
            EngineImplementation.Swap,
            EngineImplementation.Rot,
        )) {
            assertEquals(ScalarEngine.name, reached(operation, WIDE), "$operation named a vector kernel")
        }
    }

    @Test
    fun `contiguous axpy has an explicit vector path`() {
        if (!simdAvailable) return skipped()

        assertEquals(
            simdKernels.vectors.name,
            reached(EngineImplementation.Axpy, WIDE),
        )
        assertLevel1KernelsAgreeWithReference(kernels)
    }

    @Test
    fun `the standard index search observes its configured crossover`() {
        if (!simdAvailable) return skipped()
        val tuning = EngineTuning(EngineRuntime.Jvm, hostPlatform(), null)
        val tuned = standardEngine(EngineRuntime.Jvm, nativeAccess = false)
        val minimum = tuning.simdMinimum(EngineImplementation.Iamax)

        assertEquals(ScalarEngine.name, tuned.implementationFor(EngineImplementation.Iamax, minimum - 1))
        assertEquals(simdKernels.vectors.name, tuned.implementationFor(EngineImplementation.Iamax, minimum))
    }

    @Test
    fun `a strided run is scalar work whatever the operation`() {
        if (!simdAvailable) return skipped()

        for (operation in EngineImplementation.entries.takeWhile { it != EngineImplementation.IndexedDot }) {
            assertEquals(
                ScalarEngine.name,
                kernels.implementationFor(operation, WIDE, contiguous = false),
                "$operation named a vector kernel for a strided run",
            )
        }
    }

    @Test
    fun `the norm answers with its values rather than its width`() {
        if (!simdAvailable) return skipped()

        // The squared sum is tried first and abandoned when it leaves the normal range, so no kernel can be
        // named from the width alone.
        assertTrue(reached(EngineImplementation.Nrm2, WIDE) == null)
    }

    private fun skipped() {
        println("SKIPPED: the Vector API module did not resolve, so there is no arm to ask")
    }

    private companion object {
        /** Wider than any crossover here, so a kernel that exists is reached. */
        const val WIDE = 1024
    }

    @Test
    fun `simd iamax honours its contract across windows`() {
        val simd = KoblasEngine.simd ?: return
        assertIamaxHonoursItsContract(simd)
    }
}
