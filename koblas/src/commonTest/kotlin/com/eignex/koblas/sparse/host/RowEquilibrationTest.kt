package com.eignex.koblas.sparse.host

import kotlin.test.Test
import kotlin.test.assertContentEquals

class RowEquilibrationTest {

    @Test
    fun `equilibration chooses a finite power of two for each row`() {
        val rowIdx = intArrayOf(0, 0, 1, 2, 3)
        val values = doubleArrayOf(3.0, -0.75, 0.75, Double.MIN_VALUE, Double.NaN)

        val scale = equilibrationScale(4, rowIdx, values)

        assertContentEquals(doubleArrayOf(0.5, 2.0, 1.0, 1.0), scale)
    }

    @Test
    fun `scaled values use the factor belonging to each row`() {
        val rowIdx = intArrayOf(0, 1, 0)

        val scaled = scaledValues(rowIdx, doubleArrayOf(4.0, -0.25, 6.0), doubleArrayOf(0.5, 2.0))

        assertContentEquals(doubleArrayOf(2.0, -0.5, 3.0), scaled)
    }

    @Test
    fun `equilibration can be applied in place to a solve vector`() {
        val values = doubleArrayOf(3.0, 4.0)

        applyEquilibration(values, doubleArrayOf(2.0, 0.5))

        assertContentEquals(doubleArrayOf(6.0, 2.0), values)
    }
}
