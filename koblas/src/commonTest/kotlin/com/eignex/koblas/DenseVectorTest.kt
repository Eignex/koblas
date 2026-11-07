package com.eignex.koblas

import kotlin.test.*

class DenseVectorTest {
    @Test
    fun `zero builds a zero filled vector`() {
        val z = DenseVector.zero(3)
        assertEquals(3, z.size)
        for (i in 0 until 3) assertEquals(0.0, z[i])
    }

    @Test
    fun `zero rejects a negative size`() {
        assertFailsWith<DimensionMismatch> { DenseVector.zero(-3) }
    }

    @Test
    fun `of copies its input`() {
        val src = doubleArrayOf(1.0, 2.0, 3.0)
        val v = DenseVector.of(src)
        src[0] = 99.0
        assertEquals(1.0, v[0])
    }

    @Test
    fun `wrap aliases the backing array without copying`() {
        val values = doubleArrayOf(1.0, 2.0)
        val v = DenseVector.wrap(values)
        values[1] = 42.0 // mutating the source is visible through the wrapped vector
        assertEquals(42.0, v[1], 0.0)
    }
}
