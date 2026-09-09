package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import kotlin.test.*

class PreparedSparseMatrixTest {
    private fun matrix(): SparseMatrix = SparseMatrix.ofColumns(
        3,
        2,
        listOf(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0)),
    )

    @Test
    fun `a prepared matrix is an immutable snapshot`() {
        val source = matrix()
        val prepared = ReferenceSparseLinearAlgebra.prepare(source)
        source.values.fill(100.0)

        val actual = DoubleArray(3)
        prepared.gemv(1.0, doubleArrayOf(4.0, 5.0), 0.0, actual)

        assertContentEquals(doubleArrayOf(8.0, 15.0, -4.0), actual)
        prepared.close()
    }

    @Test
    fun `all prepared products agree with the reference`() {
        val source = matrix()
        ReferenceSparseLinearAlgebra.prepare(source).use { prepared ->
            val dense = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
            val expectedDense = ReferenceSparseLinearAlgebra.gemm(source, dense)
            val actualDense = DenseMatrix.zero(3, 2)
            prepared.gemm(1.0, false, dense, 0.0, actualDense)
            assertClose(expectedDense, actualDense, "dense product", tolerance = 1e-12)

            val right = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 2.0, 1 to -1.0)))
            assertEquals(ReferenceSparseLinearAlgebra.gemm(source, right), prepared.gemm(right))

            val transposedDense = DenseMatrix.wrap(2, 3, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
            val expectedTransposed = DenseMatrix.zero(2, 2)
            ReferenceSparseLinearAlgebra.gemm(
                1.0,
                source,
                true,
                transposedDense,
                true,
                0.0,
                expectedTransposed,
                right = false,
            )
            val actualTransposed = DenseMatrix.zero(2, 2)
            prepared.gemm(1.0, true, transposedDense, true, 0.0, actualTransposed, false)
            assertClose(expectedTransposed, actualTransposed, "full dense product")

            val sparseDense = DenseMatrix.zero(3, 1)
            prepared.gemm(2.0, false, right, false, 0.0, sparseDense)
            val expectedSparseDense = DenseMatrix.zero(3, 1)
            ReferenceSparseLinearAlgebra.gemm(2.0, source, false, right, false, 0.0, expectedSparseDense)
            assertClose(expectedSparseDense, sparseDense, "direct sparse dense result")
        }
    }

    @Test
    fun `prepared symmetric products retain snapshot semantics`() {
        val source = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 2.0, 1 to 3.0), listOf(1 to 5.0)),
        )
        ReferenceSparseLinearAlgebra.prepare(source).use { prepared ->
            source.values.fill(Double.NaN)
            val y = DoubleArray(2)
            prepared.symv(1.0, doubleArrayOf(7.0, 11.0), 0.0, y)
            assertContentEquals(doubleArrayOf(47.0, 76.0), y)

            val b = DenseMatrix.wrap(2, 1, doubleArrayOf(7.0, 11.0))
            val c = DenseMatrix.zero(2, 1)
            prepared.symm(1.0, b, 0.0, c)
            assertContentEquals(y, c.data)
        }
    }

    @Test
    fun `close is idempotent and rejects every product`() {
        val prepared = ReferenceSparseLinearAlgebra.prepare(matrix())
        prepared.close()
        prepared.close()

        assertFailsWith<IllegalStateException> {
            prepared.gemv(1.0, DoubleArray(2), 0.0, DoubleArray(3))
        }
        assertFailsWith<IllegalStateException> {
            prepared.gemm(1.0, false, DenseMatrix.zero(2, 1), 0.0, DenseMatrix.zero(3, 1))
        }
        assertFailsWith<IllegalStateException> {
            prepared.gemm(SparseMatrix.ofColumns(2, 0, emptyList()))
        }
        assertFailsWith<IllegalStateException> {
            prepared.symv(1.0, DoubleArray(2), 0.0, DoubleArray(2))
        }
        assertFailsWith<IllegalStateException> {
            prepared.symm(1.0, DenseMatrix.zero(2, 1), 0.0, DenseMatrix.zero(2, 1))
        }
    }
}
