package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/** What staging an aliased operand costs: the entries it addresses, in the order it addresses them. */
class StagingTest {

    @Test
    fun `a staged vector holds the entries it addresses and no more`() {
        val original = DoubleArray(64) { it.toDouble() }
        val backing = original.copyOf()
        val views = listOf(
            StridedVector(backing, 0, 64),
            StridedVector(backing, 40, 4),
            StridedVector(backing, 9, 4, 5),
            StridedVector(backing, 33, 4, -8),
        )

        for (x in views) {
            val expected = x.toDoubleArray()
            val workspace = Workspace()

            staged(workspace, x, aliased = true) { copy ->
                assertEquals(x.size, copy.values.size, "$x staged its buffer rather than its entries")
                assertContentEquals(expected, copy.toDoubleArray(), "$x")
            }

            assertEquals(1, workspace.available(x.size), "$x did not hand its loan back")
            assertContentEquals(original, backing, "$x was written through")
        }
    }
}
