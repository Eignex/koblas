package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class SparseColumnDotTest {
    @Test
    fun `column and raw timings measure the same offset support on every engine`() {
        for (rows in listOf(64, 512, 4096, 32768)) {
            val columnCase = Cases.parse("spdot-column+$rows+sparse-uniform+density=0.125").single()
            val rawCase = Cases.parse("spdot-column-raw+$rows+sparse-uniform+density=0.125").single()
            for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()) {
                val column = assertNotNull(assertNotNull(sparseArm(columnCase, engine)).work)
                val raw = assertNotNull(assertNotNull(sparseArm(rawCase, engine)).work)

                assertEquals(raw.run(), column.run())
                assertEquals("arithmetic", column.timingMode)
                assertEquals(ENGINE_COMPARISON, column.comparisonKind)
                assertEquals("${engine.name}/spdot-column", column.kernel)
                assertEquals("${engine.name}/spdot-column-raw", raw.kernel)
            }
        }
    }
}
