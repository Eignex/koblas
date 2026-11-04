package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.sparse.*
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class KoblasEngineTest {
    private val partial = object : EngineKernels {
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
            partial.dot(
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
    fun `conditional product geometry is rejected before packing`() {
        val profile = EngineProfile(
            "test",
            listOf(EngineRule("portable", setOf(EngineImplementation.Products), rows = 32..Int.MAX_VALUE)),
        )

        assertFailsWith<IllegalArgumentException> { combine(profile, BuiltinEngines.scalar) }
    }
}
