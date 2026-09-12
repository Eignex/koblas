package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class MatrixWindowTest {
    @Test
    fun `structured rectangular windows preserve parent coordinates through transpose`() {
        val data = DoubleArray(25) { it.toDouble() }
        for (structure in MatrixStructure.entries) {
            val parent = MatrixWindow(data, 5, 5, structure = structure)
            val panel = parent.window(1, 2, 2, 3)
            val transposed = panel.transpose()
            val nested = transposed.window(1, 2, 0, 2)

            for (j in 0 until 3) {
                for (i in 0 until 2) {
                    assertEquals(parent[i + 1, j + 2], panel[i, j])
                    assertEquals(panel[i, j], transposed[j, i])
                }
            }
            for (j in 0 until 2) for (i in 0 until 2) assertEquals(parent[j + 1, i + 3], nested[i, j])
        }
    }

    @Test
    fun `negative matrix strides and transpose address the backing buffer`() {
        val data = DoubleArray(20) { it.toDouble() }
        val window = MatrixWindow(data, 2, 3, offset = 17, rowStride = -2, columnStride = -5)

        val transpose = window.transpose()

        for (j in 0 until 3) {
            for (i in 0 until 2) {
                assertEquals(data[17 - 2 * i - 5 * j], window[i, j])
                assertEquals(window[i, j], transpose[j, i])
            }
        }
    }

    @Test
    fun `overlapping logical addresses are rejected`() {
        for (rowStride in listOf(-2, 2)) {
            for (columnStride in listOf(-4, 4)) {
                assertFailsWith<IllegalArgumentException> {
                    MatrixWindow(DoubleArray(30), 3, 3, 15, rowStride, columnStride)
                }
            }
        }
    }

    @Test
    fun `empty and overflowing matrix windows validate without integer wrapping`() {
        MatrixWindow(DoubleArray(0), 0, Int.MAX_VALUE, columnStride = Int.MAX_VALUE)
        assertFailsWith<IllegalArgumentException> {
            MatrixWindow(DoubleArray(4), Int.MAX_VALUE, Int.MAX_VALUE, rowStride = Int.MAX_VALUE)
        }
        assertFailsWith<IllegalArgumentException> { MatrixWindow(DoubleArray(0), 0, 0, offset = 1) }
        assertFailsWith<IllegalArgumentException> { MatrixWindow(DoubleArray(4), 2, 2, rowStride = 0) }
    }

    @Test
    fun `structured transpose preserves selected storage and implicit diagonal`() {
        val matrix = MatrixWindow(
            doubleArrayOf(Double.NaN, 2.0, 3.0, Double.NaN, Double.NaN, 4.0, Double.NaN, Double.NaN, Double.NaN),
            3,
            3,
            structure = MatrixStructure.UnitLower,
        )

        val transpose = matrix.transpose()

        for (i in 0 until 3) {
            assertEquals(1.0, transpose[i, i])
            for (j in 0 until i) assertEquals(0.0, transpose[i, j])
        }
        assertEquals(3.0, transpose[0, 2])
    }
}
