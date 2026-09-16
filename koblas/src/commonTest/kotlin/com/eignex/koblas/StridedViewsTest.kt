package com.eignex.koblas

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class StridedViewsTest {
    @Test
    fun `a negative stride presents a reversed borrowed slice`() {
        val vector = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))

        val reversed = vector.view(offset = 4, size = 3, stride = -2)
        reversed[1] = -3.0

        assertContentEquals(doubleArrayOf(5.0, -3.0, 1.0), reversed.toDoubleArray())
        assertEquals(-3.0, vector[2])
    }

    @Test
    fun `view construction rejects addresses outside the buffer`() {
        assertFailsWith<DimensionMismatch> { StridedVector(DoubleArray(3), 2, 2) }
        assertFailsWith<IllegalArgumentException> { StridedVector(DoubleArray(3), 0, 2, 0) }
    }

    @Test
    fun `a borrowed slice reaches the same kernels as dense storage`() {
        // Every second entry of the backing array, so the run the kernels walk is not contiguous.
        val backing = DoubleArray(8) { it + 1.0 }
        val even = StridedVector(backing, 0, 4, 2)
        val odd = StridedVector(backing, 1, 4, 2)

        assertEquals(1.0 * 2 + 3.0 * 4 + 5.0 * 6 + 7.0 * 8, even dot odd)
        even.scale(2.0)

        assertContentEquals(doubleArrayOf(2.0, 2.0, 6.0, 4.0, 10.0, 6.0, 14.0, 8.0), backing)
    }

    /**
     * Taking a view of a view addresses what the outer one addressed.
     *
     * `DenseVector` covers both spacings, so these are reached with a strided receiver as readily as an owned
     * one. Building the result from the buffer alone would move the window silently: every entry read would
     * still be in range and no call would fail, the answer would just be about different numbers.
     */
    @Test
    fun `a view of a strided vector composes with the window it already had`() {
        val backing = DoubleArray(12) { it.toDouble() }
        val outer: DenseVector = StridedVector(backing, offset = 2, size = 4, stride = 2)

        assertContentEquals(doubleArrayOf(2.0, 4.0, 6.0, 8.0), outer.asView().toDoubleArray())
        assertContentEquals(doubleArrayOf(4.0, 8.0), outer.view(offset = 1, size = 2, stride = 2).toDoubleArray())
    }

    /**
     * The reachable form of the same mistake: a rotation through the general overload.
     *
     * `rot(x: DenseVector, y: DenseVector, c, s)` takes a view of each operand before rotating, so a view that
     * forgot its receiver would rotate the padding instead of the window and report nothing. The run is wide
     * enough to vectorise under every platform's preferred species, so the kernel under test is the one
     * production reaches rather than a scalar tail; that kernel fuses its multiply and add, so only the
     * padding, which no arithmetic touches, is compared exactly.
     */
    @Test
    fun `a rotation over strided operands touches only the entries they address`() {
        val rng = Random(20260916)
        val n = 8
        val pad = 2
        val xs = randomVector(pad + n + pad, rng)
        val ys = randomVector(pad + n + pad, rng)
        val x0 = xs.copyOf()
        val y0 = ys.copyOf()

        // Typed as the shape that routes through asView, which is the overload that has to compose windows.
        val x: DenseVector = StridedVector(xs, pad, n)
        val y: DenseVector = StridedVector(ys, pad, n)

        rot(x, y, c = 0.6, s = 0.8)

        for (i in 0 until n) {
            assertClose(0.6 * x0[pad + i] + 0.8 * y0[pad + i], xs[pad + i], "x at $i")
            assertClose(0.6 * y0[pad + i] - 0.8 * x0[pad + i], ys[pad + i], "y at $i")
        }
        for (i in listOf(0, 1, pad + n, pad + n + 1)) {
            assertEquals(x0[i], xs[i], "x padding at $i")
            assertEquals(y0[i], ys[i], "y padding at $i")
        }
    }
}
