package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class DenseTest {
    @Test
    fun `arithmetic scaling preserves fixture magnitudes across repeated calls`() {
        val case = Cases.parse("scal+17+uniform+timing=arithmetic").single()
        val work = assertNotNull(denseWork(case, BuiltinEngines.scalar))
        val initial = Fixtures.vector(17, 1)

        repeat(101) { work.run() }

        assertEquals("arithmetic", work.timingMode)
        assertContentEquals(initial.map { -it }.toDoubleArray(), work.result)
        work.run()
        assertContentEquals(initial, work.result)
    }
}
