package com.eignex.koblas

import com.eignex.koblas.MatrixStructure
import kotlin.test.Test
import kotlin.test.assertContentEquals

class MatrixStructureTest {
    @Test
    fun `masking preserves the selected triangle across rectangular shapes`() {
        for ((rows, cols) in listOf(3 to 3, 2 to 4, 4 to 2)) {
            val backing = DoubleArray(rows * cols) { it + 1.0 }
            val matrix = DenseMatrix.wrap(rows, cols, backing)
            val expected = backing.copyOf()
            for (column in 0 until cols) {
                for (row in 0 until minOf(column, rows)) expected[row + column * rows] = 0.0
            }

            matrix.maskTo(MatrixStructure.TriangularLower)

            assertContentEquals(expected, backing, "shape ${rows}x$cols")
        }
    }
}
