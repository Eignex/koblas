package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.minus
import com.eignex.koblas.plus
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SparseCoverageCompletionTest {
    private fun selected(lower: Boolean): SparseMatrix = if (lower) {
        SparseMatrix.ofColumns(
            3,
            3,
            listOf(
                listOf(0 to 2.0, 1 to -1.0, 2 to 4.0),
                listOf(0 to Double.NaN, 1 to 3.0, 2 to 5.0),
                listOf(0 to Double.NaN, 1 to Double.NaN, 2 to 7.0),
            ),
        )
    } else {
        SparseMatrix.ofColumns(
            3,
            3,
            listOf(
                listOf(0 to 2.0, 1 to Double.NaN, 2 to Double.NaN),
                listOf(0 to -1.0, 1 to 3.0, 2 to Double.NaN),
                listOf(0 to 4.0, 1 to 5.0, 2 to 7.0),
            ),
        )
    }

    @Test
    fun `symmetric products read only the selected triangle`() {
        val x = doubleArrayOf(2.0, -3.0, 5.0)
        val expectedVector = doubleArrayOf(27.0, 14.0, 28.0)
        for (lower in booleanArrayOf(false, true)) {
            val a = selected(lower)
            val y = DoubleArray(3)
            ReferenceSparseLinearAlgebra.symv(1.0, a, x, 0.0, y, lower)
            assertContentEquals(expectedVector, y)

            val b = DenseMatrix.wrap(3, 2, doubleArrayOf(2.0, -3.0, 5.0, 11.0, 13.0, 17.0))
            val left = DenseMatrix.zero(3, 2)
            ReferenceSparseLinearAlgebra.symm(1.0, a, b, 0.0, left, lower)
            assertContentEquals(expectedVector, left.data.copyOfRange(0, 3))
            val rightInput = DenseMatrix.wrap(2, 3, doubleArrayOf(2.0, 11.0, -3.0, 13.0, 5.0, 17.0))
            val right = DenseMatrix.zero(2, 3)
            ReferenceSparseLinearAlgebra.symm(1.0, a, rightInput, 0.0, right, lower, right = true)
            for (j in 0 until 3) for (i in 0 until 2) assertEquals(left[j, i], right[i, j])
        }
    }

    @Test
    fun `symmetric products stage aliases and honor scalar no reads`() {
        val a = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(0, 1), doubleArrayOf(2.0, 5.0))
        val shared = a.values
        ReferenceSparseLinearAlgebra.symv(1.0, a, shared, 0.0, shared, lower = true)
        assertContentEquals(doubleArrayOf(4.0, 25.0), shared)

        val poisoned = SparseMatrix.wrap(1, 1, intArrayOf(0, 1), intArrayOf(0), doubleArrayOf(Double.NaN))
        val y = doubleArrayOf(Double.NaN)
        ReferenceSparseLinearAlgebra.symv(0.0, poisoned, doubleArrayOf(Double.NaN), 0.0, y)
        assertEquals(0.0, y[0])
    }

    @Test
    fun `sparse result gemm supports scaling transpose and structure`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 2.0), listOf(0 to 0.0)))
        val b = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 4.0), listOf(1 to 5.0), listOf(0 to -4.0, 1 to 6.0)))

        val actual = ReferenceSparseLinearAlgebra.gemm(-2.0, a, true, b, false)

        assertEquals(3, actual.rows)
        assertEquals(3, actual.cols)
        assertContentEquals(intArrayOf(0, 2, 3, 6), actual.copyColumnPointers())
        assertContentEquals(intArrayOf(0, 2, 1, 0, 1, 2), actual.copyRowIndices())
        assertContentEquals(doubleArrayOf(-8.0, -0.0, -20.0, 8.0, -24.0, 0.0), actual.values)

        val zero = ReferenceSparseLinearAlgebra.gemm(-0.0, a, true, b, false)
        assertContentEquals(actual.copyColumnPointers(), zero.copyColumnPointers())
        assertContentEquals(actual.copyRowIndices(), zero.copyRowIndices())
        assertTrue(zero.values.all { it.toBits() == (-0.0).toBits() })
    }

    @Test
    fun `direct dense sparse product handles every transpose and aliases`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 2.0), listOf(0 to 3.0)))
        val b = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 4.0), listOf(1 to 5.0), listOf(0 to 6.0, 1 to 7.0)))
        val expectedSparse = ReferenceSparseLinearAlgebra.gemm(1.5, a, true, b, false)
        val c = DenseMatrix.wrap(3, 3, DoubleArray(9) { 2.0 })

        ReferenceSparseLinearAlgebra.gemm(1.5, a, true, b, false, -0.5, c, Workspace())

        for (j in 0 until 3) for (i in 0 until 3) assertEquals(expectedSparse[i, j] - 1.0, c[i, j])

        val aliasedValues = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
        val aliasedA = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), aliasedValues)
        val identity = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
        val aliasedC = DenseMatrix.wrap(2, 2, aliasedValues)
        ReferenceSparseLinearAlgebra.gemm(1.0, aliasedA, false, identity, false, 0.0, aliasedC, Workspace())
        assertContentEquals(doubleArrayOf(2.0, 3.0, 5.0, 7.0), aliasedC.data)
    }

    @Test
    fun `sparse syrk provides dense and selected sparse results`() {
        val a = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 2.0), listOf(1 to 3.0, 2 to -1.0)))
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                val sparse = ReferenceSparseLinearAlgebra.syrk(a, transpose, lower)
                val n = if (transpose) a.cols else a.rows
                val dense = DenseMatrix.wrap(n, n, DoubleArray(n * n) { Double.NaN })
                ReferenceSparseLinearAlgebra.syrk(1.0, a, transpose, 0.0, dense, lower, Workspace())
                for (j in 0 until n) for (i in 0 until n) {
                    if (if (lower) i >= j else i <= j) assertEquals(sparse[i, j], dense[i, j])
                    else assertTrue(dense[i, j].isNaN())
                }
            }
        }
    }

    @Test
    fun `scaled addition retains union and subtraction direction`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
        val b = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 5.0, 1 to 7.0), listOf(1 to -3.0)))

        val sum = a + b
        val difference = a - b

        assertContentEquals(intArrayOf(0, 2, 3), sum.copyColumnPointers())
        assertContentEquals(intArrayOf(0, 1, 1), sum.copyRowIndices())
        assertContentEquals(doubleArrayOf(7.0, 7.0, 0.0), sum.values)
        assertContentEquals(doubleArrayOf(-3.0, -7.0, 6.0), difference.values)

        a.values.fill(Double.NaN)
        val zero = ReferenceSparseLinearAlgebra.addScaled(-0.0, a, false, b)
        assertContentEquals(intArrayOf(0, 2, 3), zero.copyColumnPointers())
        assertEquals(5.0, zero[0, 0])
        assertEquals(7.0, zero[1, 0])
        assertEquals(-3.0, zero[1, 1])
    }
}
