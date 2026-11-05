package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.engine.Vendor
import com.eignex.koblas.koblas
import kotlin.test.*

class ExactArmRejectionTest {
    private fun backend(operations: Set<EngineImplementation>) = object : KoblasEngine {
        override val name = "openblas"
        override val vendor = Vendor.OpenBlas
        override val implementations = operations
    }

    @Test
    fun `declared vendor exports qualify for exact timing`() {
        val binding = backend(setOf(EngineImplementation.Gemm))
        assertNull(exactArmRejection(binding, EngineImplementation.Gemm))
    }

    @Test
    fun `missing exports are rejected before timing`() {
        val binding = backend(emptySet())
        assertNotNull(exactArmRejection(binding, EngineImplementation.Gemmt))
    }
}
