package com.eignex.koblas.sparse

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparseProductTest {

    private fun sparseAndDense(
        rows: Int,
        cols: Int,
        rng: Random,
        density: Double = 0.35,
    ): Pair<F64SparseMatrix, F64DenseMatrix> {
        val dense = F64DenseMatrix.zero(rows, cols)
        val columns = ArrayList<List<Pair<Int, Double>>>(cols)
        for (j in 0 until cols) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until rows) {
                if (rng.nextDouble() >= density) continue
                val v = rng.nextDouble(-1.0, 1.0)
                column.add(i to v)
                dense[i, j] = v
            }
            columns.add(column)
        }
        return F64SparseMatrix.ofColumns(rows, cols, columns) to dense
    }

    private fun copyOf(m: F64DenseMatrix) = F64DenseMatrix.wrap(m.rows, m.cols, m.data.copyOf())

    @Test
    fun `every transpose combination agrees with the dense product`() {
        val rng = Random(20260826)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val (sparse, dense) = sparseAndDense(5, 3, rng)
                val m = if (transposeA) 3 else 5
                val k = if (transposeA) 5 else 3
                val b = if (transposeB) randomMatrix(4, k, rng) else randomMatrix(k, 4, rng)
                val c = randomMatrix(m, 4, rng)

                val fromDense = copyOf(c)
                koblas.gemm(0.75, dense, transposeA, b, transposeB, -0.5, fromDense)
                val fromSparse = copyOf(c)
                koblas.sparseBlas.gemm(0.75, sparse, transposeA, b, transposeB, -0.5, fromSparse)

                assertClose(fromDense, fromSparse, "transposeA=$transposeA transposeB=$transposeB")
            }
        }
    }

    @Test
    fun `one right-hand side agrees with gemv`() {
        val rng = Random(20260827)
        val (sparse, _) = sparseAndDense(6, 4, rng)
        val x = randomVector(4, rng)
        val c = F64DenseMatrix.zero(6, 1)

        koblas.sparseBlas.gemm(1.0, sparse, false, F64DenseMatrix.wrap(4, 1, x.copyOf()), false, 0.0, c)

        assertClose(sparse.gemv(x), c.data, "gemm over one column is gemv")
    }

    @Test
    fun `a beta of zero overwrites a destination it never reads`() {
        val rng = Random(20260828)
        val (sparse, _) = sparseAndDense(4, 3, rng)
        val b = randomMatrix(3, 2, rng)
        val c = F64DenseMatrix.wrap(4, 2, DoubleArray(8) { Double.NaN })

        koblas.sparseBlas.gemm(1.0, sparse, false, b, false, 0.0, c)

        assertTrue(c.data.all { it.isFinite() }, "a NaN survived beta = 0")
    }

    @Test
    fun `an alpha of zero leaves the destination scaled by beta alone`() {
        val rng = Random(20260829)
        val (sparse, _) = sparseAndDense(4, 3, rng)
        val b = randomMatrix(3, 2, rng)
        val c = randomMatrix(4, 2, rng)
        val expected = F64DenseMatrix.wrap(4, 2, DoubleArray(8) { c.data[it] * 2.0 })

        val actual = copyOf(c)
        koblas.sparseBlas.gemm(0.0, sparse, false, b, false, 2.0, actual)

        assertClose(expected, actual, "alpha = 0")
    }

    @Test
    fun `the product of a matrix and an identity is the matrix`() {
        val rng = Random(20260830)
        val (sparse, dense) = sparseAndDense(5, 4, rng)

        assertClose(dense, sparse.gemm(F64DenseMatrix.diagonal(4)), "A times I")
    }

    @Test
    fun `the convenience overload takes its shape from the operands`() {
        val rng = Random(20260831)
        val (sparse, _) = sparseAndDense(5, 4, rng)

        val c = sparse.gemm(randomMatrix(4, 3, rng))

        assertEquals(5, c.rows)
        assertEquals(3, c.cols)
    }

    @Test
    fun `a destination of the wrong shape is rejected`() {
        val rng = Random(20260901)
        val (sparse, _) = sparseAndDense(5, 4, rng)
        val b = randomMatrix(4, 3, rng)

        assertFailsWith<DimensionMismatch> {
            koblas.sparseBlas.gemm(1.0, sparse, false, b, false, 0.0, F64DenseMatrix.zero(5, 2))
        }
    }

    @Test
    fun `a second operand that does not meet the first is rejected`() {
        val rng = Random(20260902)
        val (sparse, _) = sparseAndDense(5, 4, rng)

        assertFailsWith<DimensionMismatch> { sparse.gemm(randomMatrix(3, 3, rng)) }
    }
}
