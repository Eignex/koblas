package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.PortableEngine
import com.eignex.koblas.internal.jvm25.simd.SparseSimd
import com.eignex.koblas.internal.jvm25.simd.simdAvailable
import com.eignex.koblas.internal.jvm25.simd.simdContributions
import kotlin.test.Test
import kotlin.test.assertNotEquals

private val selectedBackend = BuiltinEngines.simd ?: PortableEngine

class SimdIndexedSparseKernelsTest {
    @Test
    fun `the indexed norm is never an exact vector measurement`() {
        if (!simdAvailable) return skipped()
        val kernels = selectedBackend

        val reached = kernels.implementationFor(EngineImplementation.IndexedNrm2, WIDE, true)

        // Gathering may retry with scalar rescaling; hosts without gather use scalar from the start.
        assertNotEquals(simdContributions.first().name, reached)
    }

    @Test
    fun `a support narrower than a lane block never names the vector kernel`() {
        if (!simdAvailable) return skipped()
        val kernels = selectedBackend

        for (count in 1 until SparseSimd.lanes) {
            for (operation in EngineImplementation.entries.filter {
                it.ordinal in
                    EngineImplementation.IndexedDot.ordinal until
                    EngineImplementation.Panels.ordinal
            }) {
                assertNotEquals(
                    simdContributions.first().name,
                    kernels.implementationFor(operation, count, true),
                    "$operation at $count",
                )
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
