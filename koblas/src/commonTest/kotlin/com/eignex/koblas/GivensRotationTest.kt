package com.eignex.koblas

import com.eignex.koblas.core.F64DenseVector
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class GivensRotationTest {

    @Test
    fun `rotg generates the rotation that zeroes the second component`() {
        val pairs = listOf(3.0 to 4.0, -3.0 to 4.0, 3.0 to -4.0, 1.0 to 0.0, 0.0 to 1.0, -2.0 to 0.0)
        for ((a, b) in pairs) {
            val g = rotg(a, b)
            assertEquals(sqrt(a * a + b * b), abs(g.r), 1e-12, "r is not the pair's length for ($a, $b)")
            assertEquals(1.0, g.c * g.c + g.s * g.s, 1e-12, "not a rotation for ($a, $b)")
            assertEquals(g.r, g.c * a + g.s * b, 1e-12, "rotation does not produce r for ($a, $b)")
            assertEquals(0.0, g.c * b - g.s * a, 1e-12, "rotation does not zero b for ($a, $b)")
        }
    }

    @Test
    fun `rotg on an all-zero pair is the identity rotation`() {
        val g = rotg(0.0, 0.0)
        assertEquals(1.0, g.c)
        assertEquals(0.0, g.s)
        assertEquals(0.0, g.r)
    }

    @Test
    fun `rotg survives components that square out of range`() {
        val g = rotg(3e200, 4e200)
        assertEquals(5e200, abs(g.r), 1e188)
        assertEquals(1.0, g.c * g.c + g.s * g.s, 1e-12)
        val tiny = rotg(3e-200, 4e-200)
        assertEquals(5e-200, abs(tiny.r), 1e-212)
        assertEquals(1.0, tiny.c * tiny.c + tiny.s * tiny.s, 1e-12)
    }

    @Test
    fun `rot applies the rotation elementwise and preserves length`() {
        val rng = Random(20260810)
        val n = 7
        val x = F64DenseVector.of(randomVector(n, rng))
        val y = F64DenseVector.of(randomVector(n, rng))
        val x0 = x.data.copyOf()
        val y0 = y.data.copyOf()
        val g = rotg(2.0, 1.0)

        rot(x, y, g)
        for (i in 0 until n) {
            assertEquals(g.c * x0[i] + g.s * y0[i], x.data[i], 1e-12, "x at $i")
            assertEquals(g.c * y0[i] - g.s * x0[i], y.data[i], 1e-12, "y at $i")
        }
        for (i in 0 until n) {
            val before = x0[i] * x0[i] + y0[i] * y0[i]
            val after = x.data[i] * x.data[i] + y.data[i] * y.data[i]
            assertEquals(before, after, 1e-12, "length changed at $i")
        }
    }

    @Test
    fun `rotg and rot together zero the target entry`() {
        val x = F64DenseVector.of(doubleArrayOf(3.0, 1.0, 2.0))
        val y = F64DenseVector.of(doubleArrayOf(4.0, -1.0, 0.5))
        val g = rotg(x[0], y[0])
        rot(x, y, g)
        assertEquals(0.0, y[0], 1e-12, "the leading entry of y was not eliminated")
        assertEquals(5.0, x[0], 1e-12)
    }
}
