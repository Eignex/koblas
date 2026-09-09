package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GemmtTest {
    @Test
    fun `gemmt agrees with scalar oracle in every orientation`() {
        val n = 5
        val k = 7
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                for (lower in booleanArrayOf(false, true)) {
                    val a = if (transposeA) DenseMatrix(k, n) else DenseMatrix(n, k)
                    val b = if (transposeB) DenseMatrix(n, k) else DenseMatrix(k, n)
                    a.data.indices.forEach { a.data[it] = (it % 11 - 5) / 7.0 }
                    b.data.indices.forEach { b.data[it] = (it % 13 - 6) / 9.0 }
                    val initial = DoubleArray(n * n) { (it - 4) / 8.0 }
                    val actual = DenseMatrix.wrap(n, n, initial.copyOf())
                    val expected = initial.copyOf()
                    for (j in 0 until n) {
                        for (i in 0 until n) {
                            if (if (lower) i >= j else i <= j) {
                                var sum = 0.0
                                for (p in 0 until k) {
                                    val av = if (transposeA) a[p, i] else a[i, p]
                                    val bv = if (transposeB) b[j, p] else b[p, j]
                                    sum += av * bv
                                }
                                expected[i + j * n] = 1.25 * sum - 0.5 * expected[i + j * n]
                            }
                        }
                    }

                    ReferenceBlas.gemmt(1.25, a, transposeA, b, transposeB, -0.5, actual, lower, Workspace())

                    assertClose(expected, actual.data, "tA=$transposeA tB=$transposeB lower=$lower", 1e-11)
                }
            }
        }
    }

    @Test
    fun `gemmt preserves the unselected triangle and zero no read rules`() {
        for (lower in booleanArrayOf(false, true)) {
            val a = DenseMatrix.wrap(2, 3, DoubleArray(6) { Double.NaN })
            val b = DenseMatrix.wrap(3, 2, DoubleArray(6) { Double.POSITIVE_INFINITY })
            val c = DenseMatrix.wrap(2, 2, DoubleArray(4) { Double.NaN })

            ReferenceBlas.gemmt(0.0, a, false, b, false, 0.0, c, lower)

            for (j in 0 until 2) {
                for (i in 0 until 2) {
                    if (if (lower) i >= j else i <= j) {
                        assertEquals(0.0, c[i, j])
                    } else {
                        assertTrue(c[i, j].isNaN())
                    }
                }
            }
        }
    }

    @Test
    fun `gemmt stages either aliased input`() {
        val original = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
        val other = DenseMatrix.wrap(2, 2, doubleArrayOf(11.0, 13.0, 17.0, 19.0))
        for (aliasA in booleanArrayOf(false, true)) {
            val shared = DenseMatrix.wrap(2, 2, original.copyOf())
            val a = if (aliasA) shared else other
            val b = if (aliasA) other else shared
            val expected = shared.data.copyOf()
            val aCopy = a.data.copyOf()
            val bCopy = b.data.copyOf()
            for (j in 0 until 2) {
                for (i in j until 2) {
                    var sum = 0.0
                    for (p in 0 until 2) sum += aCopy[i + p * 2] * bCopy[p + j * 2]
                    expected[i + j * 2] = sum
                }
            }

            ReferenceBlas.gemmt(1.0, a, false, b, false, 0.0, shared, workspace = Workspace())

            assertClose(expected, shared.data, "aliasA=$aliasA")
        }
    }

    @Test
    fun `gemmt exceptional path evaluates stored dense products`() {
        val a = DenseMatrix.wrap(1, 1, doubleArrayOf(Double.POSITIVE_INFINITY))
        val b = DenseMatrix.wrap(1, 1, doubleArrayOf(0.0))
        val c = DenseMatrix.zero(1)

        ReferenceBlas.gemmt(1.0, a, false, b, false, 0.0, c)

        assertTrue(c[0, 0].isNaN())
    }

    @Test
    fun `gemmt nontransposed path scales B before multiplying A`() {
        val a = DenseMatrix.wrap(2, 1, doubleArrayOf(1e308, Double.NaN))
        val b = DenseMatrix.wrap(1, 2, doubleArrayOf(2.0, 2.0))
        val c = DenseMatrix.zero(2)

        ReferenceBlas.gemmt(1e-308, a, false, b, false, 0.0, c)

        assertEquals(2.0, c[0, 0], 1e-15)
        assertTrue(c[1, 0].isNaN())
    }

    @Test
    fun `gemmt transposed path scales the completed overflowing dot`() {
        val a = DenseMatrix.wrap(1, 1, doubleArrayOf(1e308))
        val b = DenseMatrix.wrap(1, 1, doubleArrayOf(2.0))
        val c = DenseMatrix.zero(1)

        ReferenceBlas.gemmt(1e-308, a, true, b, false, 0.0, c)

        assertEquals(Double.POSITIVE_INFINITY, c[0, 0])
    }

    @Test
    fun `gemmt transposed path retains overflow before cancellation`() {
        val a = DenseMatrix.wrap(2, 1, doubleArrayOf(1e308, 1e308))
        val b = DenseMatrix.wrap(2, 1, doubleArrayOf(2.0, -2.0))
        val c = DenseMatrix.zero(1)

        ReferenceBlas.gemmt(1e-308, a, true, b, false, 0.0, c)

        assertTrue(c[0, 0].isNaN())
    }

    @Test
    fun `gemmt zero depth applies beta in every orientation`() {
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val a = if (transposeA) DenseMatrix(0, 2) else DenseMatrix(2, 0)
                val b = if (transposeB) DenseMatrix(2, 0) else DenseMatrix(0, 2)
                val c = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

                ReferenceBlas.gemmt(Double.NaN, a, transposeA, b, transposeB, -2.0, c)

                assertContentEquals(doubleArrayOf(-2.0, -4.0, 3.0, -8.0), c.data)
            }
        }
    }
}
