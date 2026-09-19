package com.eignex.koblas

import com.eignex.koblas.*
import com.eignex.koblas.dense.ScalarVectorKernels
import kotlin.test.*

class OperatorsTest {

    private val a = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
    private val b = DenseMatrix.ofRows(arrayOf(doubleArrayOf(5.0, 6.0), doubleArrayOf(7.0, 8.0)))
    private val x = DenseVector.of(doubleArrayOf(2.0, -1.0))

    @Test
    fun `scalar operators preserve the logical entries of a view`() {
        for (size in intArrayOf(0, 3)) {
            for (stride in intArrayOf(1, 2, -2)) {
                val backing = DoubleArray(9) { it + 0.5 }
                val original = backing.copyOf()
                val view = StridedVector(backing, if (stride < 0) 6 else 1, size, stride)
                val expected = view.toDoubleArray()
                ScalarVectorKernels.scale(expected, 0, 2.0, size)

                val right = view * 2.0
                val left = 2.0 * view
                val negated = -view

                assertVectorAgreesWithReference(expected, right)
                assertVectorAgreesWithReference(expected, left)
                ScalarVectorKernels.scale(expected, 0, -0.5, size)
                assertVectorAgreesWithReference(expected, negated)
                assertContentEquals(original, backing)
            }
        }
    }

    @Test
    fun `vector sums and differences use each operands logical spacing`() {
        for (size in intArrayOf(0, 3)) {
            for (stride in intArrayOf(1, 2, -2)) {
                val first = DoubleArray(9) { it + 0.5 }
                val second = DoubleArray(7) { it - 0.25 }
                val originalFirst = first.copyOf()
                val originalSecond = second.copyOf()
                val x = StridedVector(first, if (stride < 0) 6 else 1, size, stride)
                val y = StridedVector(second, 5, size, -2)
                for (alpha in doubleArrayOf(1.0, -1.0)) {
                    val expected = x.toDoubleArray()
                    ScalarVectorKernels.axpy(expected, 0, alpha, y.toDoubleArray(), 0, size)

                    val actual = if (alpha == 1.0) x + y else x - y

                    assertVectorAgreesWithReference(expected, actual)
                    assertContentEquals(originalFirst, first)
                    assertContentEquals(originalSecond, second)
                }
            }
        }
    }

    private fun assertVectorAgreesWithReference(expected: DoubleArray, actual: DenseVector) {
        assertClose(expected, actual.toDoubleArray(), "vector operator")
    }

    @Test
    fun `matrix product has expected entries`() {
        assertEquals(DenseMatrix.ofRows(arrayOf(doubleArrayOf(19.0, 22.0), doubleArrayOf(43.0, 50.0))), a * b)
    }

    @Test
    fun `matrix-vector product agrees with gemv`() {
        assertEquals(DenseVector.wrap(koblas.gemv(a, x.values)), a * x)
        assertClose(doubleArrayOf(0.0, 2.0), (a * x).values, "a * x")
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
        assertClose(doubleArrayOf(2.5, 3.0), (x + y).values, "x + y")
        assertClose(doubleArrayOf(1.5, -5.0), (x - y).values, "x - y")
        assertTrue(doubleArrayOf(2.0, -1.0).contentEquals(x.values), "x should be untouched")
    }

    @Test
    fun `scalar multiplication commutes and negation is the -1 case`() {
        assertEquals(a * 2.0, 2.0 * a)
        assertEquals(a * -1.0, -a)
        assertEquals(x * 2.0, 2.0 * x)
        assertEquals(x * -1.0, -x)
        assertEquals(6.0, (a * 2.0)[1, 0])
        assertClose(doubleArrayOf(-2.0, 1.0), (-x).values, "-x")
    }

    @Test
    fun `mismatched shapes are rejected`() {
        val wide = DenseMatrix.zero(2, 3)
        assertFailsWith<DimensionMismatch> { a + wide }
        assertFailsWith<DimensionMismatch> { a - wide }
        assertFailsWith<DimensionMismatch> { x + DenseVector.zero(3) }
        assertFailsWith<DimensionMismatch> { x - DenseVector.zero(3) }
    }
}
