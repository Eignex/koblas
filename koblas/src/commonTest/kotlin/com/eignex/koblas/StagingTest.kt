package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertSame

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

    /**
     * What a seam taking one array is given: the vector's own array where it is the whole of one, and a
     * workspace loan otherwise. The pass-through is the case worth pinning, because a gather there would be
     * a copy of every ordinary operand.
     */
    @Test
    fun `a vector spanning its array is passed through and any other spacing is gathered from the workspace`() {
        val backing = DoubleArray(64) { it.toDouble() }
        val workspace = Workspace()

        contiguous(workspace, DenseVector.wrap(backing)) { values ->
            assertSame(backing, values, "a vector spanning its own array was copied")
        }
        assertEquals(0, workspace.idleLengths(), "a pass-through took a loan")

        for (x in listOf(StridedVector(backing, 40, 4), StridedVector(backing, 33, 4, -8))) {
            val expected = x.toDoubleArray()

            contiguous(workspace, x) { values -> assertContentEquals(expected, values, "$x") }

            assertEquals(1, workspace.available(x.size), "$x did not gather from the workspace")
        }
    }
}
