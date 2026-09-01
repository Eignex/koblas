package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.test.*

class F64LuDecompositionTest {

    @Test
    fun `rectangular LU reconstructs tall and wide matrices`() {
        val matrices = listOf(
            F64DenseMatrix.of(arrayOf(doubleArrayOf(0.0, 2.0), doubleArrayOf(3.0, 4.0), doubleArrayOf(5.0, 6.0))),
            F64DenseMatrix.of(arrayOf(doubleArrayOf(0.0, 2.0, 7.0), doubleArrayOf(3.0, 4.0, 8.0))),
        )
        for (a in matrices) {
            val factor = F64ReferenceLinearAlgebra.factor(a)
            assertEquals(a.rows, factor.rows)
            assertEquals(a.cols, factor.cols)
            assertEquals(minOf(a.rows, a.cols), factor.order)
            assertEquals(factor.order, factor.n, "legacy n is the pivot order")
            assertEquals((0 until a.rows).toList(), factor.permutation().sorted())
            assertClose(
                product(factor.lower(), factor.upper()).data,
                permutedRows(a, factor.rowPermutation).data,
                "${a.rows}x${a.cols}",
            )
        }
    }

    @Test
    fun `rectangular LU reports rank deficiency and empty shapes`() {
        val deficient = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(2.0, 4.0, 6.0)),
        ).let(F64ReferenceLinearAlgebra::factor)
        assertTrue(deficient.rankDeficient)
        assertTrue(deficient.singular, "legacy singular reports a zero DGETRF pivot")
        assertEquals(1, deficient.rank)

        for ((rows, cols) in listOf(0 to 0, 0 to 3, 3 to 0)) {
            val factor = F64ReferenceLinearAlgebra.factor(F64DenseMatrix(rows, cols))
            assertEquals(0, factor.order)
            assertEquals(0, factor.rank)
            assertFalse(factor.rankDeficient)
            assertEquals(rows * cols, factor.lu.size)
            assertEquals(rows, factor.rowPermutation.size)
        }
    }

    @Test
    fun `factorInto reuses rectangular buffers and requires the exact shape`() {
        val first = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0), doubleArrayOf(5.0, 6.0)),
        )
        val out = F64ReferenceLinearAlgebra.factor(first)
        val factors = out.lu
        val pivots = out.mutablePivots
        val second = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(0.0, 2.0), doubleArrayOf(3.0, 4.0), doubleArrayOf(5.0, 6.0)),
        )
        assertSame(out, F64ReferenceLinearAlgebra.factorInto(second, out))
        assertSame(factors, out.lu)
        assertSame(pivots, out.mutablePivots)
        assertClose(F64ReferenceLinearAlgebra.factor(second).lu, out.lu, "rectangular factorInto")
        assertFailsWith<DimensionMismatch> {
            F64ReferenceLinearAlgebra.factorInto(F64DenseMatrix(2, 3), out)
        }
    }

    @Test
    fun `square only LU operations reject rectangular factors`() {
        val a = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)),
        )
        val factor = F64ReferenceLinearAlgebra.factor(a)
        assertFailsWith<DimensionMismatch> { factor.determinant() }
        assertFailsWith<DimensionMismatch> { F64ReferenceLinearAlgebra.solve(factor, DoubleArray(a.rows)) }
        assertFailsWith<DimensionMismatch> { F64ReferenceLinearAlgebra.invert(factor) }
        assertFailsWith<DimensionMismatch> { F64ReferenceLinearAlgebra.rcond(factor, a.norm1()) }
    }

    private fun permutedRows(a: F64DenseMatrix, piv: IntArray): F64DenseMatrix =
        F64DenseMatrix(a.rows, a.cols).also { p ->
            for (j in 0 until a.cols) for (i in 0 until a.rows) p[i, j] = a[piv[i], j]
        }

    private fun product(a: F64DenseMatrix, b: F64DenseMatrix): F64DenseMatrix =
        F64DenseMatrix(a.rows, b.cols).also { c ->
            for (j in 0 until b.cols) {
                for (k in 0 until a.cols) {
                    for (i in 0 until a.rows) {
                        c[i, j] += a[i, k] * b[k, j]
                    }
                }
            }
        }
}
