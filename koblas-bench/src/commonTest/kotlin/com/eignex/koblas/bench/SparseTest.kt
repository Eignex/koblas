package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertNotNull

class SparseTest {
    @Test
    fun `gather timing variants return the same values across repeated calls`() {
        val resetCase = Cases.parse("spgather+64+sparse-uniform+density=0.25").single()
        val arithmeticCase = Cases.parse("${resetCase.id}+timing=arithmetic").single()
        val reset = assertNotNull(sparseWork(resetCase, BuiltinEngines.scalar))
        val arithmetic = assertNotNull(sparseWork(arithmeticCase, BuiltinEngines.scalar))

        repeat(3) {
            reset.run()
            arithmetic.run()
            assertContentEquals(reset.result, arithmetic.result)
        }
    }
}
