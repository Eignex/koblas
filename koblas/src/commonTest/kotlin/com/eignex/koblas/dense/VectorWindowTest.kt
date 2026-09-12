package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class VectorWindowTest {
    @Test
    fun `negative vector stride retains its source`() {
        val data = doubleArrayOf(1.0, 2.0, 3.0)
        val vector = VectorWindow(data, 2, 2, -2)
        data[0] = 4.0
        assertEquals(3.0, vector[0])
        assertEquals(4.0, vector[1])
    }

    @Test
    fun `vector bounds use wide arithmetic`() {
        assertFailsWith<IllegalArgumentException> { VectorWindow(DoubleArray(1), Int.MAX_VALUE, 0, Int.MIN_VALUE) }
        assertFailsWith<IllegalArgumentException> { VectorWindow(DoubleArray(1), 2) }
        assertFailsWith<IllegalArgumentException> { VectorWindow(DoubleArray(1), 0, stride = 0) }
        VectorWindow(DoubleArray(0), 0, stride = -1)
    }
}
