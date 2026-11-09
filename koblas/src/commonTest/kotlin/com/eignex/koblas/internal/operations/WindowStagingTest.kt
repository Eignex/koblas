package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.storage.windowsOverlap
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
                scratch,
                DenseVector.wrap(backing, offset, count, stride),
                aliased = true,
            ) { copy ->
                assertEquals(count, copy.size, "$offset/$count/$stride staged its buffer rather than its entries")
                assertContentEquals(expected, copy.toDoubleArray())
            }

            assertEquals(1, scratch.available(count), "$offset/$count/$stride did not hand its loan back")
            assertContentEquals(original, backing, "$offset/$count/$stride was written through")
        }
    }

    @Test
    fun `a disjoint window is passed through without scratch`() {
        val backing = DoubleArray(64)
        val scratch = Scratch()

        val view = DenseVector.wrap(backing, 33, 4, -8)
        stagedWindow(scratch, view, aliased = false) { stable ->
            assertSame(view, stable)
        }

        assertEquals(0, scratch.idleLengths())
    }

    @Test
    fun `identical mappings share one snapshot`() {
        for ((offset, count, stride) in listOf(Triple(0, 4, 1), Triple(7, 4, -2), Triple(2, 1, 3))) {
            val backing = DoubleArray(8) { it.toDouble() }
            val destination = DenseMatrix.wrap(4, 2, backing)
            val x = DenseVector.wrap(backing, offset, count, stride)
            val y = DenseVector.wrap(backing, offset, count, if (count == 1) 1 else stride)
            val expected = x.toDoubleArray()
            val scratch = Scratch()

            stagedWindows(scratch, x, y, destination) { sx, sy ->
                backing.fill(-1.0)
                assertSame(sx, sy)
                assertContentEquals(expected, sx.toDoubleArray())
            }

            assertEquals(1, scratch.available(count))
        }
    }

    @Test
    fun `aliased vector calls reuse their scratch`() {
        for (symmetric in booleanArrayOf(false, true)) {
            val a = DenseMatrix.wrap(3, 3, DoubleArray(9) { 1.0 + it })
            val workspace = Scratch()
            val actual = DenseVector.wrap(doubleArrayOf(1.0, 2.0, 3.0))
            val expected = actual.toDoubleArray()

            repeat(3) {
                val input = expected.copyOf()
                if (symmetric) {
                    ReferenceBlas.symv(0.5, a, input, 0.25, expected)
                    KoblasEngine.scalar.symv(0.5, a, actual, 0.25, actual, scratch = workspace)
                } else {
                    ReferenceBlas.gemv(0.5, a, input, 0.25, expected)
                    KoblasEngine.scalar.gemv(0.5, a, actual, 0.25, actual, scratch = workspace)
                }
            }

            assertClose(expected, actual.toDoubleArray(), "aliased matrix vector symmetric=$symmetric")
            assertEquals(1, workspace.available(actual.size))
            assertEquals(1, workspace.idleLengths())
        }
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
