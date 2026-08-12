package com.eignex.koblas

import com.eignex.koblas.dense.matMul
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class OperatorsTest {

    private val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
    private val b = DenseMatrix.of(arrayOf(doubleArrayOf(5.0, 6.0), doubleArrayOf(7.0, 8.0)))
    private val x = DenseVector.of(doubleArrayOf(2.0, -1.0))

    @Test
    fun `matrix product agrees with matMul`() {
        assertEquals(a.matMul(b), a * b)
    }

    @Test
    fun `matrix-vector product agrees with gemv`() {
        assertEquals(DenseVector.wrap(koblas.gemv(a, x.data)), a * x)
        assertClose(doubleArrayOf(0.0, 2.0), (a * x).data, "a * x")
    }

    @Test
    fun `sparse matrix-vector product agrees with the sparse gemv`() {
        val s = SparseMatrix.ofTriplets(
            rows = 2,
            cols = 2,
            rowIdx = intArrayOf(0, 0, 1, 1),
            colIdx = intArrayOf(0, 1, 0, 1),
            values = doubleArrayOf(1.0, 2.0, 3.0, 4.0),
        )
        assertEquals(a * x, s * x, "the sparse and dense spellings of the same matrix should agree")
    }

    @Test
    fun `sum and difference agree with axpy on a copy`() {
        val sum = a + b
        val difference = a - b
        for (i in 0 until 2) {
            for (j in 0 until 2) {
                assertEquals(a[i, j] + b[i, j], sum[i, j], "sum ($i,$j)")
                assertEquals(a[i, j] - b[i, j], difference[i, j], "difference ($i,$j)")
            }
        }
        assertEquals(1.0, a[0, 0])
        assertEquals(5.0, b[0, 0])
    }

    @Test
    fun `vector sum and difference agree entrywise`() {
        val y = DenseVector.of(doubleArrayOf(0.5, 4.0))
        assertClose(doubleArrayOf(2.5, 3.0), (x + y).data, "x + y")
        assertClose(doubleArrayOf(1.5, -5.0), (x - y).data, "x - y")
        assertTrue(doubleArrayOf(2.0, -1.0).contentEquals(x.data), "x should be untouched")
    }

    @Test
    fun `scalar multiplication commutes and negation is the -1 case`() {
        assertEquals(a * 2.0, 2.0 * a)
        assertEquals(a * -1.0, -a)
        assertEquals(x * 2.0, 2.0 * x)
        assertEquals(x * -1.0, -x)
        assertEquals(6.0, (a * 2.0)[1, 0])
        assertClose(doubleArrayOf(-2.0, 1.0), (-x).data, "-x")
    }

    @Test
    fun `mismatched shapes are rejected`() {
        val wide = DenseMatrix.zero(2, 3)
        assertFailsWith<IllegalArgumentException> { a + wide }
        assertFailsWith<IllegalArgumentException> { a - wide }
        assertFailsWith<IllegalArgumentException> { x + DenseVector.zero(3) }
        assertFailsWith<IllegalArgumentException> { x - DenseVector.zero(3) }
    }
}
