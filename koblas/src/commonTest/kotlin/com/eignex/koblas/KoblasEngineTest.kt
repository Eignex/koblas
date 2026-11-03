package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.sparse.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class KoblasEngineTest {
    private val partial = object : KoblasEngine {
        override val name = "partial"
        override val implementations = setOf(EngineImplementation.Dot)
        override fun dot(
            a: DoubleArray,
            aOff: Int,
            b: DoubleArray,
            bOff: Int,
            len: Int,
            aStride: Int,
            bStride: Int,
        ): Double {
            var sum = 0.0
            for (i in 0 until len) sum += a[aOff + i * aStride] * b[bOff + i * bStride]
            return sum
        }
    }

    @Test
    fun `a partial provider never inherits portable arithmetic`() {
        assertFailsWith<UnsupportedOperationException> { partial.sum(doubleArrayOf(1.0), 0, 1) }
        assertFailsWith<UnsupportedOperationException> {
            partial.dotDense(
                intArrayOf(0),
                0,
                doubleArrayOf(1.0),
                0,
                1,
                doubleArrayOf(1.0),
            )
        }
        assertFailsWith<UnsupportedOperationException> { partial.tileRows }
        assertNull(partial.implementationFor(EngineImplementation.Sum, 1))
    }

    @Test
    fun `composition calls separate providers for supported and missing parts`() {
        val engine = KoblasEngine.combine(
            EngineProfile("partial with portable", listOf(EngineRule(partial.id, partial.implementations))),
            partial,
            BuiltinEngines.scalar,
        )
        assertEquals(setOf(partial.id, BuiltinEngines.scalar.id), engine.providers.map { it.id }.toSet())
        val values = doubleArrayOf(0.5, -2.0, 3.0)

        val dot = engine.dot(values, 0, values, 0, values.size)
        val sum = engine.sum(values, 0, values.size)

        assertArithmeticAgreesWithReference(BuiltinEngines.scalar.dot(values, 0, values, 0, values.size), dot)
        assertArithmeticAgreesWithReference(BuiltinEngines.scalar.sum(values, 0, values.size), sum)
        assertEquals(partial.name, engine.implementationFor(EngineImplementation.Dot, values.size))
        assertEquals(BuiltinEngines.scalar.name, engine.implementationFor(EngineImplementation.Sum, values.size))
    }

    private fun assertArithmeticAgreesWithReference(expected: Double, actual: Double) =
        assertClose(expected, actual, "provider composition")
}
