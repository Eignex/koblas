package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Vector
import com.eignex.koblas.koblas
import com.eignex.koblas.syr
import com.eignex.koblas.syr2
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.fail

/**
 * Zero multipliers must leave operands unread. Poisoned [Vector] implementations make reads
 * observable even when the numerical contribution would be zero.
 */
class SparseNoReadTest {

    /** A [Vector] whose entries cannot be read, only counted. */
    private class PoisonVector(override val size: Int) : Vector {
        override fun get(i: Int): Double = fail("a vector this call must not read was read at $i")
        override fun toDoubleArray(): DoubleArray = fail("a vector this call must not read was materialised")
    }

    private fun example(): SparseMatrix = SparseMatrix.ofColumns(
        2,
        2,
        listOf(listOf(0 to 2.0, 1 to 3.0), listOf(1 to 5.0)),
    )

    @Test
    fun `a zero multiplier rank one update copies the source without reading the vector`() {
        val source = example()

        val result = source.syr(0.0, PoisonVector(2))

        assertEquals(source, result)
    }

    @Test
    fun `a zero multiplier rank two update copies the source without reading either vector`() {
        val source = example()

        val result = source.syr2(0.0, PoisonVector(2), PoisonVector(2))

        assertEquals(source, result)
    }

    @Test
    fun `a rank update result owns its arrays whatever the multiplier`() {
        val source = example()

        val result = source.syr(0.0, PoisonVector(2))
        source.values.fill(Double.NaN)

        assertContentEquals(doubleArrayOf(2.0, 3.0, 5.0), result.values)
    }
}
