package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class VectorViewsTest {
    @Test
    fun `view cases verify the full buffer and use the default policy`() {
        for (line in listOf(
            "copy-overlap+17+uniform", "axpy-views+17+uniform", "swap-views+17+uniform",
            "spdot-view+17+sparse-uniform+density=0.1", "spdot-view+17+sparse-uniform+density=1.0",
        )) {
            val case = Cases.parse(line).single()
            val work = assertNotNull(vectorViewsArm(case, koblas).work)

            assertEquals("default-policy", work.comparisonKind)
            assertTrue(assertNotNull(work.kernel).startsWith("portable-vector+"))
            repeat(3) { assertTrue(work.run().isFinite()) }
            for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).filter { it !== koblas }) {
                assertNull(vectorViewsArm(case, engine).work)
            }
        }
    }
}
