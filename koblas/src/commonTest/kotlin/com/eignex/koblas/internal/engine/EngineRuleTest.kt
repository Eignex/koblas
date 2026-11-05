package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.engine.EngineImplementation.*
import com.eignex.koblas.internal.kernels.EngineKernels
import kotlin.test.*

class EngineRuleTest {
    @Test
    fun `matrix preferences bound minimum extents and the borrowed layout`() {
        val backend = provider("matrix", setOf(Gemm))
        val rule = EngineRule(backend.id, backend.implementations, 2, 3, 6, borrowed = false)
        val engine = combine(EngineTuning("bounded", listOf(rule)), backend)

        for (dimension in 0..2) {
            val low = intArrayOf(2, 3, 6)
            low[dimension]--
            assertNull(engine.wholeCallFor(Gemm, low[0], low[1], low[2], false))
        }
        assertSame(backend, engine.wholeCallFor(Gemm, 2, 3, 6, false))
        assertSame(backend, engine.wholeCallFor(Gemm, 4, 5, 8, false))
        assertNull(engine.wholeCallFor(Gemm, 2, 3, 6, true))
    }

    @Test
    fun `work boundaries avoid overflow at extreme extents`() {
        val rule = EngineRule("test", setOf(Gemm), work = Long.MAX_VALUE)

        assertTrue(rule.matches(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE, true, false))
        assertFalse(rule.matches(Int.MAX_VALUE, 1, Int.MAX_VALUE, true, false))
        assertFalse(rule.matches(0, Int.MAX_VALUE, Int.MAX_VALUE, true, false))
    }

    private fun provider(id: String, operations: Set<EngineImplementation>): KoblasEngine = object : EngineKernels {
        override val name = id
        override val implementations = operations
    }
}
