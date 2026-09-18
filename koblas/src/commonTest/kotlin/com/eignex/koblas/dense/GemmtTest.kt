package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GemmtTest {
    @Test
    fun `gemmt agrees with the scalar oracle in every orientation`() = withDenseBlas { blas ->
        val n = 5
        val k = 7
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                for (lower in booleanArrayOf(false, true)) {
                    val a = if (transposeA) DenseMatrix(k, n) else DenseMatrix(n, k)
                    val b = if (transposeB) DenseMatrix(n, k) else DenseMatrix(k, n)
                    a.values.indices.forEach { a.values[it] = (it % 11 - 5) / 7.0 }
                    b.values.indices.forEach { b.values[it] = (it % 13 - 6) / 9.0 }
                    val initial = DoubleArray(n * n) { (it - 4) / 8.0 }
                    val actual = DenseMatrix.wrap(n, n, initial.copyOf())
                    val expected = DenseMatrix.wrap(n, n, initial.copyOf())
                    ReferenceBlas.gemmt(1.25, a, transposeA, b, transposeB, -0.5, expected, lower)

                    blas.gemmt(1.25, a, transposeA, b, transposeB, -0.5, actual, lower)

                    assertClose(expected.values, actual.values, "tA=$transposeA tB=$transposeB lower=$lower", 1e-11)
                }
            }
        }
    }

    @Test
    fun `gemmt preserves the unselected triangle and zero no read rules`() = withDenseBlas { blas ->
        for (lower in booleanArrayOf(false, true)) {
            val a = DenseMatrix.wrap(2, 3, DoubleArray(6) { Double.NaN })
            val b = DenseMatrix.wrap(3, 2, DoubleArray(6) { Double.POSITIVE_INFINITY })
            val c = DenseMatrix.wrap(2, 2, DoubleArray(4) { Double.NaN })

            blas.gemmt(0.0, a, false, b, false, 0.0, c, lower)

            for (j in 0 until 2) {
                for (i in 0 until 2) {
                    if (if (lower) i >= j else i <= j) {
                        assertEquals(0.0, c[i, j])
                    } else {
                        assertTrue(c[i, j].isNaN(), "the unselected triangle at ($i, $j) was written")
                    }
                }
            }
        }
    }

    @Test
    fun `gemmt zero depth applies beta in every orientation`() = withDenseBlas { blas ->
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val a = if (transposeA) DenseMatrix(0, 2) else DenseMatrix(2, 0)
                val b = if (transposeB) DenseMatrix(2, 0) else DenseMatrix(0, 2)
                val c = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

                blas.gemmt(Double.NaN, a, transposeA, b, transposeB, -2.0, c)

                assertContentEquals(doubleArrayOf(-2.0, -4.0, 3.0, -8.0), c.values)
            }
        }
    }
}
