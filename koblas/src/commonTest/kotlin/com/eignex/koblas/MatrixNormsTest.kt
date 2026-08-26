package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MatrixNormsTest {

    private val example = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(1.0, -2.0, 3.0),
            doubleArrayOf(-4.0, 5.0, -6.0),
        ),
    )

    @Test
    fun `normInf and normFro agree with direct sums`() {
        assertEquals(15.0, example.normInf(), 1e-12)
        assertEquals(sqrt(1.0 + 4 + 9 + 16 + 25 + 36), example.normFro(), 1e-12)

        val empty = F64DenseMatrix(0, 0)
        assertEquals(0.0, empty.normInf())
        assertEquals(0.0, empty.normFro())
    }

    @Test
    fun `normInf through a reused workspace repeats its answer`() {
        val ws = Workspace().apply { reserve(example.rows, count = 2) }
        assertEquals(example.normInf(), example.normInf(ws), 1e-12)
        assertEquals(example.normInf(), example.normInf(ws), 1e-12)
    }

    /**
     * `dlange` carries a NaN out through `DISNAN`. Tracking the maximum by comparison alone would drop it,
     * since every comparison against a NaN is false, and answer for the columns that happen to be clean.
     */
    @Test
    fun `norm1 and normInf carry a NaN through`() {
        val poisoned = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, -2.0, 3.0),
                doubleArrayOf(-4.0, Double.NaN, -6.0),
            ),
        )

        assertTrue(poisoned.norm1().isNaN(), "norm1 dropped the NaN")
        assertTrue(poisoned.normInf().isNaN(), "normInf dropped the NaN")
    }

    @Test
    fun `a NaN in one column does not hide behind a larger clean column`() {
        // The clean column sums to 100, so a maximum that merely compares would answer with it.
        val poisoned = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(Double.NaN, 100.0),
                doubleArrayOf(1.0, 0.0),
            ),
        )

        assertTrue(poisoned.norm1().isNaN(), "norm1 answered from the clean column")
        assertTrue(poisoned.normInf().isNaN(), "normInf answered from the clean row")
    }

    @Test
    fun `normFro survives entries that square out of range`() {
        val big = F64DenseMatrix.of(arrayOf(doubleArrayOf(3e200, 4e200)))
        assertEquals(5e200, big.normFro(), 1e188)
    }
}
