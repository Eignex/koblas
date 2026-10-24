package com.eignex.koblas

import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class CholeskyFactorTest {
    private fun referenceProduct(
        left: DenseMatrix,
        right: DenseMatrix,
        transposeA: Boolean = false,
        transposeB: Boolean = false,
    ): DenseMatrix {
        val result = DenseMatrix.zero(
            if (transposeA) left.cols else left.rows,
            if (transposeB) right.rows else right.cols,
        )
        BuiltinEngines.scalar.gemm(1.0, left, transposeA, right, transposeB, 0.0, result)
        return result
    }

    @Test
    fun `factorization reads only its selected triangle and agrees with scalar kernels`() {
        val random = Random(71)
        for (n in listOf(0, 1, 3, 17)) {
            val source = wellConditioned(n, random)
            val expected = referenceProduct(source, source, transposeB = true)
            val reference = factorCholesky(expected, true, BuiltinEngines.scalar).lowerFactor()
            for (lower in listOf(true, false)) {
                val input = expected.copyOf()
                for (i in 0 until n) {
                    for (j in 0 until n) {
                        if ((lower && i < j) || (!lower && i > j)) input[i, j] = Double.NaN
                    }
                }
                val before = input.values.copyOf()
                val actual = input.cholesky(lower).lowerFactor()
                assertFactorAgreesWithReference(reference, actual)
                assertFactorAgreesWithReference(expected, referenceProduct(actual, actual, transposeB = true))
                assertEquals(before.toList(), input.values.toList())
            }
        }
    }

    private fun assertFactorAgreesWithReference(expected: DenseMatrix, actual: DenseMatrix) {
        assertClose(expected, actual, "Cholesky factor or reconstruction", 1e-11)
    }

    @Test
    fun `vector and matrix solves recover right hand sides without changing inputs`() {
        val source = wellConditioned(5, Random(72))
        val matrix = referenceProduct(source, source, transposeB = true)
        val factor = matrix.cholesky()
        val right = randomMatrix(5, 3, Random(73))
        val before = right.copyOf()
        assertFactorAgreesWithReference(right, referenceProduct(matrix, factor.solve(right)))
        assertFactorAgreesWithReference(before, right)
        val vector = right.toArray().map { it[0] }.toDoubleArray()
        assertClose(vector, BuiltinEngines.scalar.gemv(matrix, factor.solve(vector)), "vector solve", 1e-11)
        val lower = factor.lowerFactor()
        for (transpose in listOf(false, true)) {
            val solved = factor.solveLower(right, transpose)
            assertFactorAgreesWithReference(right, referenceProduct(lower, solved, transposeA = transpose))
        }
    }

    @Test
    fun `factor snapshots remain independent of inputs and returned factors`() {
        val matrix = DenseMatrix.diagonal(doubleArrayOf(2.0, 3.0))
        val factor = matrix.cholesky()
        val expected = factor.solve(doubleArrayOf(1.0, 1.0)).toList()
        matrix[0, 0] = 99.0
        factor.lowerFactor()[0, 0] = 99.0
        assertEquals(expected, factor.solve(doubleArrayOf(1.0, 1.0)).toList())
        assertClose(ln(6.0), factor.logDeterminant(), "log determinant")
        assertEquals(0.0, DenseMatrix.zero(0).cholesky().logDeterminant())
    }

    @Test
    fun `nonpositive and nonfinite pivots identify the failing position`() {
        for (pivot in listOf(0.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY)) {
            val failure = assertFailsWith<NotPositiveDefinite> {
                DenseMatrix.diagonal(doubleArrayOf(1.0, pivot)).cholesky()
            }
            assertEquals(1, failure.position)
        }
    }

    @Test
    fun `nonsquare matrices and incompatible solves reject dimensions`() {
        assertFailsWith<DimensionMismatch> { DenseMatrix.zero(2, 3).cholesky() }
        val factor = DenseMatrix.diagonal(2).cholesky()
        assertFailsWith<DimensionMismatch> { factor.solve(doubleArrayOf(1.0)) }
        assertFailsWith<DimensionMismatch> { factor.solve(DenseMatrix.zero(1, 2)) }
    }
}
