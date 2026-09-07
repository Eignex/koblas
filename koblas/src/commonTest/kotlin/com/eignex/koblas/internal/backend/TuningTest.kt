package com.eignex.koblas.internal.backend

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The resolution rule the dense and sparse tuning collections share, exercised on the decision itself
 * rather than through a process whose environment a test would have to rewrite.
 */
class TuningTest {
    @Test
    fun `an absent override keeps the default`() {
        val resolved = tunedIntValue(null, default = 128, minimum = 1, maximum = Int.MAX_VALUE)

        assertEquals(128, resolved)
    }

    @Test
    fun `a blank override keeps the default`() {
        for (configured in listOf("", "   ", "\t")) {
            val resolved = tunedIntValue(configured, default = 128, minimum = 1, maximum = Int.MAX_VALUE)

            assertEquals(128, resolved, "blank override [$configured]")
        }
    }

    @Test
    fun `an integer override replaces the default`() {
        val resolved = tunedIntValue("256", default = 128, minimum = 1, maximum = Int.MAX_VALUE)

        assertEquals(256, resolved)
    }

    @Test
    fun `surrounding space does not stop an override parsing`() {
        val resolved = tunedIntValue("  256  ", default = 128, minimum = 1, maximum = Int.MAX_VALUE)

        assertEquals(256, resolved)
    }

    @Test
    fun `an unparseable override keeps the default`() {
        for (configured in listOf("many", "12.5", "0x40", "9999999999", "64,128", "-")) {
            val resolved = tunedIntValue(configured, default = 128, minimum = 1, maximum = Int.MAX_VALUE)

            assertEquals(128, resolved, "unparseable override [$configured]")
        }
    }

    @Test
    fun `an override below the minimum keeps the default`() {
        for (configured in listOf("0", "-1", "-4096")) {
            val resolved = tunedIntValue(configured, default = 128, minimum = 1, maximum = Int.MAX_VALUE)

            assertEquals(128, resolved, "override below the minimum [$configured]")
        }
    }

    @Test
    fun `an override above the maximum keeps the default`() {
        val resolved = tunedIntValue("65", default = 64, minimum = 1, maximum = Long.SIZE_BITS)

        assertEquals(64, resolved)
    }

    @Test
    fun `an override at either end of the range is accepted`() {
        assertEquals(1, tunedIntValue("1", default = 64, minimum = 1, maximum = Long.SIZE_BITS))
        assertEquals(64, tunedIntValue("64", default = 8, minimum = 1, maximum = Long.SIZE_BITS))
    }

    @Test
    fun `a fractional override replaces the default within the unit interval`() {
        assertEquals(0.25, tunedDoubleValue("0.25", default = 0.1, minimum = 0.0, maximum = 1.0))
        assertEquals(0.0, tunedDoubleValue("0", default = 0.1, minimum = 0.0, maximum = 1.0))
        assertEquals(1.0, tunedDoubleValue("1", default = 0.1, minimum = 0.0, maximum = 1.0))
    }

    @Test
    fun `a fractional override outside the range keeps the default`() {
        for (configured in listOf("-0.5", "1.5", "nonsense", "NaN", "")) {
            val resolved = tunedDoubleValue(configured, default = 0.1, minimum = 0.0, maximum = 1.0)

            assertEquals(0.1, resolved, "override outside the range [$configured]")
        }
    }

    @Test
    fun `a key spells one property and one environment variable`() {
        assertEquals("koblas.dense.jvm.c.dot4.crossover", tuningProperty("dense", "jvm.c.dot4.crossover"))
        assertEquals("KOBLAS_DENSE_JVM_C_DOT4_CROSSOVER", tuningEnvironment("dense", "jvm.c.dot4.crossover"))
        assertEquals("koblas.sparse.dot.dense.c.crossover", tuningProperty("sparse", "dot.dense.c.crossover"))
        assertEquals("KOBLAS_SPARSE_DOT_DENSE_C_CROSSOVER", tuningEnvironment("sparse", "dot.dense.c.crossover"))
    }
}
