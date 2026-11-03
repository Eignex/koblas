package com.eignex.koblas.internal.operands

import com.eignex.koblas.*
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

class WindowStagingTest {
    @Test
    fun `a staged window holds the entries it addresses and no more`() {
        val original = DoubleArray(64) { it.toDouble() }
        val backing = original.copyOf()
        for ((offset, count, stride) in listOf(
            Triple(0, 64, 1),
            Triple(40, 4, 1),
            Triple(9, 4, 5),
            Triple(33, 4, -8),
        )) {
            val expected = DoubleArray(count) { backing[offset + it * stride] }
            val scratch = Scratch()

            stagedWindow(
                scratch.arrays,
                backing,
                offset,
                count,
                stride,
                aliased = true,
            ) { copy, origin, step ->
                assertEquals(count, copy.size, "$offset/$count/$stride staged its buffer rather than its entries")
                assertContentEquals(expected, DoubleArray(count) { copy[origin + it * step] })
            }

            assertEquals(1, scratch.available(count), "$offset/$count/$stride did not hand its loan back")
            assertContentEquals(original, backing, "$offset/$count/$stride was written through")
        }
    }

    @Test
    fun `a disjoint window is passed through without scratch`() {
        val backing = DoubleArray(64)
        val scratch = Scratch()

        stagedWindow(scratch.arrays, backing, 33, 4, -8, aliased = false) { values, offset, stride ->
            assertSame(backing, values)
            assertEquals(33, offset)
            assertEquals(-8, stride)
        }

        assertEquals(0, scratch.idleLengths())
    }

    @Test
    fun `windows overlap only where they share an addressed entry`() {
        val backing = DoubleArray(32)
        val other = DoubleArray(32)

        assertTrue(windowsOverlap(backing, 0, 8, 2, backing, 4, 4, 2))
        assertFalse(windowsOverlap(backing, 0, 8, 2, backing, 1, 8, 2))
        assertTrue(windowsOverlap(backing, 31, 4, -3, backing, 22, 2, 1))
        assertFalse(windowsOverlap(backing, 0, 8, 1, other, 0, 8, 1))
    }
}
