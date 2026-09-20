package com.eignex.koblas.sparse

import com.eignex.koblas.dense.simdAvailable
import kotlin.test.Test
import kotlin.test.assertNotEquals

class SimdIndexedSparseKernelsTest {
    @Test
    fun `the indexed norm is never an exact vector measurement`() {
        if (!simdAvailable) return skipped()
        val kernels = SimdIndexedSparseKernels

        val reached = kernels.implementationFor(SparseOperation.IndexedNrm2, WIDE)

        // Gathering may retry with scalar rescaling; hosts without gather use scalar from the start.
        assertNotEquals(kernels.name, reached)
    }

    @Test
    fun `a support narrower than a lane block never names the vector kernel`() {
        if (!simdAvailable) return skipped()
        val kernels = SimdIndexedSparseKernels

        for (count in 1 until SparseSimd.lanes) {
            for (operation in SparseOperation.entries) {
                assertNotEquals(kernels.name, kernels.implementationFor(operation, count), "$operation at $count")
            }
        }
    }

    private fun skipped() {
        println("SKIPPED: the Vector API module did not resolve, so there is no indexed arm to ask")
    }

    private companion object {
        const val WIDE = 1 shl 12
    }
}
