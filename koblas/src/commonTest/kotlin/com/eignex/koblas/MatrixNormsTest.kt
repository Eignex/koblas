package com.eignex.koblas

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals

class MatrixNormsTest {

    private val example = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(1.0, -2.0, 3.0),
            doubleArrayOf(-4.0, 5.0, -6.0),
        ),
    )

    @Test
    fun `normInf and normFro agree with direct sums`() {
        assertEquals(15.0, normInf(example), 1e-12)
        assertEquals(sqrt(1.0 + 4 + 9 + 16 + 25 + 36), normFro(example), 1e-12)

        val empty = F64DenseMatrix(0, 0)
        assertEquals(0.0, normInf(empty))
        assertEquals(0.0, normFro(empty))
    }

    @Test
    fun `normInf through a reused workspace repeats its answer`() {
        val ws = Workspace().apply { reserve(example.rows, count = 2) }
        assertEquals(normInf(example), normInf(example, ws), 1e-12)
        assertEquals(normInf(example), normInf(example, ws), 1e-12)
    }

    @Test
    fun `normFro survives entries that square out of range`() {
        val big = F64DenseMatrix.of(arrayOf(doubleArrayOf(3e200, 4e200)))
        assertEquals(5e200, normFro(big), 1e188)
    }
}
