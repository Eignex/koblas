package com.eignex.koblas.internal.storage.sparse

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class SparseRowSumsTest {
    @Test
    fun `updates and growth preserve every accumulated row`() {
        val keys = listOf(0, Int.MAX_VALUE - 1, Int.MAX_VALUE) + List(65) { it * 8192 + 4096 }
        val expected = mutableMapOf<Int, Double>()
        val sums = SparseRowSums()
        val random = Random(17)

        repeat(3) {
            for (row in keys.shuffled(random)) {
                val value = random.nextInt(-32, 65) * 0.125
                expected[row] = (expected[row] ?: 0.0) + value

                sums.add(row, value)

                assertEquals(maxOf(0.0, expected.values.maxOrNull() ?: 0.0), sums.maximumOrZero())
            }
        }
    }

    @Test
    fun `absent rows contribute zero to the maximum`() {
        val sums = SparseRowSums()
        assertEquals(0.0, sums.maximumOrZero())

        sums.add(Int.MAX_VALUE, -3.0)

        assertEquals(0.0, sums.maximumOrZero())
    }

    @Test
    fun `negative rows are rejected before modifying sums`() {
        val sums = SparseRowSums()
        sums.add(4, 2.0)

        assertFailsWith<IllegalArgumentException> { sums.add(-1, 9.0) }

        assertEquals(2.0, sums.maximumOrZero())
    }
}
