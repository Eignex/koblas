package com.eignex.koblas

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * The plane rotation, which is all of Givens that Koblas keeps.
 *
 * The generator, BLAS `drotg`, is gone: it turned two scalars into the cosine and sine below plus a rotated
 * length only it used, with no vectorised path and no caller anywhere. The rotations here are therefore
 * written out, and `(0.6, 0.8)` is the one `drotg(3, 4)` produced.
 */
class RotTest {
    private val c = 0.6
    private val s = 0.8

    @Test
    fun `rot applies the rotation elementwise and preserves length`() {
        val rng = Random(20260810)
        val n = 7
        val x = DenseVector.of(randomVector(n, rng))
        val y = DenseVector.of(randomVector(n, rng))
        val x0 = x.values.copyOf()
        val y0 = y.values.copyOf()

        rot(x, y, c, s)

        for (i in 0 until n) {
            assertEquals(c * x0[i] + s * y0[i], x.values[i], 1e-12, "x at $i")
            assertEquals(c * y0[i] - s * x0[i], y.values[i], 1e-12, "y at $i")
        }
        for (i in 0 until n) {
            val before = x0[i] * x0[i] + y0[i] * y0[i]
            val after = x.values[i] * x.values[i] + y.values[i] * y.values[i]
            assertEquals(before, after, 1e-12, "length changed at $i")
        }
    }

    /** What the routine is for: a rotation built from the leading pair eliminates the second of them. */
    @Test
    fun `a rotation from the leading pair zeroes the target entry`() {
        val x = DenseVector.of(doubleArrayOf(3.0, 1.0, 2.0))
        val y = DenseVector.of(doubleArrayOf(4.0, -1.0, 0.5))

        rot(x, y, c, s)

        assertEquals(0.0, y[0], 1e-12, "the leading entry of y was not eliminated")
        assertEquals(5.0, x[0], 1e-12)
    }

    @Test
    fun `the identity rotation leaves both operands alone`() {
        val x = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val y = DenseVector.of(doubleArrayOf(Double.NaN, Double.NaN))

        rot(x, y, c = 1.0, s = 0.0)

        assertContentEquals(doubleArrayOf(1.0, 2.0), x.values, "the identity wrote through a NaN operand")
    }

    @Test
    fun `the rot kernel rotates the offset window and leaves the rest alone`() {
        val x = DoubleArray(7) { it + 1.0 }
        val y = DoubleArray(7) { 10.0 * (it + 1) }
        val expectedX = x.copyOf()
        val expectedY = y.copyOf()
        // The two runs start at different offsets, so x(2 + i) pairs with y(3 + i).
        for (i in 0 until 3) {
            expectedX[2 + i] = c * x[2 + i] + s * y[3 + i]
            expectedY[3 + i] = c * y[3 + i] - s * x[2 + i]
        }

        koblas.vectorKernels.rot(x, 2, y, 3, 3, c, s)

        for (i in x.indices) {
            assertEquals(expectedX[i], x[i], 1e-12, "x[$i]")
            assertEquals(expectedY[i], y[i], 1e-12, "y[$i]")
        }
    }

    @Test
    fun `rot supports strided inputs and snapshots overlaps`() {
        val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val x = StridedVector(backing, 0, 4)
        val y = StridedVector(backing, 4, 4, -1)
        val originalX = x.toDoubleArray()
        val originalY = y.toDoubleArray()
        val expected = backing.copyOf()
        for (i in 0 until x.size) {
            expected[x.offset + i * x.stride] = c * originalX[i] + s * originalY[i]
        }
        for (i in 0 until y.size) {
            expected[y.offset + i * y.stride] = c * originalY[i] - s * originalX[i]
        }

        rot(x, y, c, s)

        assertContentEquals(expected, backing)
    }
}
