package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.prepare
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class PreparedSparseMatrixTest {
    private fun matrix(): SparseMatrix = SparseMatrix.ofColumns(
        3,
        2,
        listOf(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0)),
    )

    @Test
    fun `a prepared matrix is an immutable snapshot`() {
        val source = matrix()
        val prepared = koblas.prepare(source)
        source.values.fill(100.0)

        val actual = DoubleArray(3)
        prepared.gemv(1.0, doubleArrayOf(4.0, 5.0), 0.0, actual)

        assertContentEquals(doubleArrayOf(8.0, 15.0, -4.0), actual)
    }

    @Test
    fun `the prepared shape reports the snapshot`() {
        val prepared = matrix().prepare()

        assertEquals(3, prepared.rows)
        assertEquals(2, prepared.cols)
        assertEquals(3, prepared.nnz)
    }

    @Test
    fun `all prepared products agree with the one-shot calls`() {
        val source = matrix()
        val prepared = koblas.prepare(source)
        val dense = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val expectedDense = koblas.gemm(source, dense)
        val actualDense = DenseMatrix.zero(3, 2)
        prepared.gemm(1.0, false, dense, 0.0, actualDense)
        assertClose(expectedDense, actualDense, "dense product")

        val right = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 2.0, 1 to -1.0)))
        assertEquals(koblas.gemm(source, right), prepared.gemm(right))

        val transposedDense = DenseMatrix.wrap(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expectedTransposed = DenseMatrix.zero(2, 2)
        koblas.gemm(1.0, source, true, transposedDense, true, 0.0, expectedTransposed, right = false)
        val actualTransposed = DenseMatrix.zero(2, 2)
        prepared.gemm(1.0, true, transposedDense, true, 0.0, actualTransposed, false)
        assertClose(expectedTransposed, actualTransposed, "full dense product")

        val sparseDense = DenseMatrix.zero(3, 1)
        prepared.gemm(2.0, false, right, false, 0.0, sparseDense)
        val expectedSparseDense = DenseMatrix.zero(3, 1)
        koblas.gemm(2.0, source, false, right, false, 0.0, expectedSparseDense)
        assertClose(expectedSparseDense, sparseDense, "direct sparse dense result")
    }

    @Test
    fun `a prepared transposed product agrees with the one-shot transpose`() {
        val source = matrix()
        val prepared = source.prepare()
        val b = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.5, 2 to -2.0), listOf(1 to 0.5)))

        // Twice, because the second call is the one that reuses the derived transpose rather than deriving it.
        repeat(2) {
            assertEquals(koblas.gemm(1.0, source, true, b, false), prepared.gemm(1.0, true, b, false))
        }
    }

    @Test
    fun `prepared symmetric products retain snapshot semantics`() {
        val source = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 3.0), listOf(1 to 5.0)))
        val prepared = koblas.prepare(source)
        source.values.fill(Double.NaN)
        val y = DoubleArray(2)
        prepared.symv(1.0, doubleArrayOf(7.0, 11.0), 0.0, y)
        assertContentEquals(doubleArrayOf(47.0, 76.0), y)

        val b = DenseMatrix.wrap(2, 1, doubleArrayOf(7.0, 11.0))
        val c = DenseMatrix.zero(2, 1)
        prepared.symm(1.0, b, 0.0, c)
        assertContentEquals(y, c.values)
    }

    @Test
    fun `a prepared snapshot keeps its own structural arrays`() {
        val pointers = intArrayOf(0, 1, 2)
        val rows = intArrayOf(0, 1)
        val values = doubleArrayOf(2.0, 3.0)
        val source = SparseMatrix.wrap(2, 2, pointers, rows, values)
        val prepared = source.prepare()

        values[0] = Double.NaN
        val y = DoubleArray(2)
        prepared.gemv(1.0, doubleArrayOf(1.0, 1.0), 0.0, y)

        assertContentEquals(doubleArrayOf(2.0, 3.0), y)
    }
}
