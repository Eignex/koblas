package com.eignex.koblas.sparse

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.test.*

class F64PreparedSparseMatrixTest {
    private fun matrix(): F64SparseMatrix = F64SparseMatrix.ofColumns(
        3,
        2,
        listOf(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0)),
    )

    @Test
    fun `a prepared matrix is an immutable snapshot`() {
        val source = matrix()
        val prepared = F64ReferenceSparseLinearAlgebra.prepare(source)
        source.values.fill(100.0)

        val actual = DoubleArray(3)
        prepared.gemv(1.0, doubleArrayOf(4.0, 5.0), 0.0, actual)

        assertContentEquals(doubleArrayOf(8.0, 15.0, -4.0), actual)
        prepared.close()
    }

    @Test
    fun `all prepared products agree with the reference`() {
        val source = matrix()
        F64ReferenceSparseLinearAlgebra.prepare(source).use { prepared ->
            val dense = F64DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
            val expectedDense = F64ReferenceSparseLinearAlgebra.gemm(source, dense)
            val actualDense = F64DenseMatrix.zero(3, 2)
            prepared.gemm(1.0, false, dense, 0.0, actualDense)
            assertClose(expectedDense, actualDense, "dense product", tolerance = 1e-12)

            val right = F64SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 2.0, 1 to -1.0)))
            assertEquals(F64ReferenceSparseLinearAlgebra.gemm(source, right), prepared.gemm(right))
        }
    }

    @Test
    fun `close is idempotent and rejects every product`() {
        val prepared = F64ReferenceSparseLinearAlgebra.prepare(matrix())
        prepared.close()
        prepared.close()

        assertFailsWith<IllegalStateException> {
            prepared.gemv(1.0, DoubleArray(2), 0.0, DoubleArray(3))
        }
        assertFailsWith<IllegalStateException> {
            prepared.gemm(1.0, false, F64DenseMatrix.zero(2, 1), 0.0, F64DenseMatrix.zero(3, 1))
        }
        assertFailsWith<IllegalStateException> {
            prepared.gemm(F64SparseMatrix.ofColumns(2, 0, emptyList()))
        }
    }
}
