package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.minus
import com.eignex.koblas.plus
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SparseBlasTest {
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
            ReferenceSparseBlas.symv(1.0, a, x, 0.0, y, lower)
            assertContentEquals(expectedVector, y)

            val b = DenseMatrix.wrap(3, 2, doubleArrayOf(2.0, -3.0, 5.0, 11.0, 13.0, 17.0))
            val left = DenseMatrix.zero(3, 2)
            ReferenceSparseBlas.symm(1.0, a, b, 0.0, left, lower)
            assertContentEquals(expectedVector, left.data.copyOfRange(0, 3))
            val rightInput = DenseMatrix.wrap(2, 3, doubleArrayOf(2.0, 11.0, -3.0, 13.0, 5.0, 17.0))
            val right = DenseMatrix.zero(2, 3)
            ReferenceSparseBlas.symm(1.0, a, rightInput, 0.0, right, lower, right = true)
            for (j in 0 until 3) for (i in 0 until 2) assertEquals(left[j, i], right[i, j])
        }
    }

    @Test
    fun `symmetric products stage aliases and honor scalar no reads`() {
        val a = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(0, 1), doubleArrayOf(2.0, 5.0))
        val shared = a.values
        ReferenceSparseBlas.symv(1.0, a, shared, 0.0, shared, lower = true)
        assertContentEquals(doubleArrayOf(4.0, 25.0), shared)

        val poisoned = SparseMatrix.wrap(1, 1, intArrayOf(0, 1), intArrayOf(0), doubleArrayOf(Double.NaN))
        val y = doubleArrayOf(Double.NaN)
        ReferenceSparseBlas.symv(0.0, poisoned, doubleArrayOf(Double.NaN), 0.0, y)
        assertEquals(0.0, y[0])

        val allShared = doubleArrayOf(2.0, 3.0, 7.0, 5.0)
        val sharedA = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), allShared)
        val sharedB = DenseMatrix.wrap(2, 2, allShared)
        val sharedC = DenseMatrix.wrap(2, 2, allShared)
        ReferenceSparseBlas.symm(
            1.0,
            sharedA,
            sharedB,
            0.0,
            sharedC,
            lower = true,
            workspace = Workspace(),
        )
        assertContentEquals(doubleArrayOf(13.0, 21.0, 29.0, 46.0), sharedC.data)
    }

    @Test
    fun `sparse result gemm supports scaling transpose and structure`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 2.0), listOf(0 to 0.0)))
        val b = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 4.0), listOf(1 to 5.0), listOf(0 to -4.0, 1 to 6.0)))

        val actual = ReferenceSparseBlas.gemm(-2.0, a, true, b, false)

        assertEquals(3, actual.rows)
        assertEquals(3, actual.cols)
        assertContentEquals(intArrayOf(0, 2, 3, 6), actual.copyColumnPointers())
        assertContentEquals(intArrayOf(0, 2, 1, 0, 1, 2), actual.copyRowIndices())
        assertContentEquals(doubleArrayOf(-8.0, -0.0, -20.0, 8.0, -24.0, 0.0), actual.values)

        val zero = ReferenceSparseBlas.gemm(-0.0, a, true, b, false)
        assertContentEquals(actual.copyColumnPointers(), zero.copyColumnPointers())
        assertContentEquals(actual.copyRowIndices(), zero.copyRowIndices())
        assertTrue(zero.values.all { it.toBits() == (-0.0).toBits() })
    }

    @Test
    fun `direct dense sparse product handles every transpose and aliases`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 2.0), listOf(0 to 3.0)))
        val b = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 4.0), listOf(1 to 5.0), listOf(0 to 6.0, 1 to 7.0)))
        val expectedSparse = ReferenceSparseBlas.gemm(1.5, a, true, b, false)
        val c = DenseMatrix.wrap(3, 3, DoubleArray(9) { 2.0 })

        ReferenceSparseBlas.gemm(1.5, a, true, b, false, -0.5, c, Workspace())

        for (j in 0 until 3) for (i in 0 until 3) assertEquals(expectedSparse[i, j] - 1.0, c[i, j])

        val aliasedValues = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
        val aliasedA = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), aliasedValues)
        val identity = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
        val aliasedC = DenseMatrix.wrap(2, 2, aliasedValues)
        ReferenceSparseBlas.gemm(1.0, aliasedA, false, identity, false, 0.0, aliasedC, Workspace())
        assertContentEquals(doubleArrayOf(2.0, 3.0, 5.0, 7.0), aliasedC.data)
    }

    @Test
    fun `sparse syrk provides dense and selected sparse results`() {
        val a = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 2.0), listOf(1 to 3.0, 2 to -1.0)))
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                val sparse = ReferenceSparseBlas.syrk(a, transpose, lower)
                val n = if (transpose) a.cols else a.rows
                val dense = DenseMatrix.wrap(n, n, DoubleArray(n * n) { Double.NaN })
                ReferenceSparseBlas.syrk(1.0, a, transpose, 0.0, dense, lower, Workspace())
                for (j in 0 until n) {
                    for (i in 0 until n) {
                        if (if (lower) i >= j else i <= j) {
                            assertEquals(sparse[i, j], dense[i, j])
                        } else {
                            assertTrue(dense[i, j].isNaN())
                        }
                    }
                }
            }
        }

        val exceptional = SparseMatrix.ofColumns(
            2,
            1,
            listOf(listOf(0 to Double.POSITIVE_INFINITY, 1 to 0.0)),
        )
        val exceptionalResult = ReferenceSparseBlas.syrk(exceptional)
        assertTrue(exceptionalResult[1, 0].isNaN())
        val implicit = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to Double.POSITIVE_INFINITY)))
        val implicitDense = DenseMatrix.zero(2)
        ReferenceSparseBlas.syrk(1.0, implicit, false, 0.0, implicitDense)
        assertEquals(0.0, implicitDense[1, 0], "implicit sparse zero must not form zero times infinity")
    }

    @Test
    fun `sparse syrk follows diagonal stored adjacency in both orientations`() {
        val n = 1024
        val diagonal = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j % 7 + 1.0)) })

        for (transpose in booleanArrayOf(false, true)) {
            val result = ReferenceSparseBlas.syrk(diagonal, transpose)
            assertEquals(n, result.nnz)
            assertContentEquals(IntArray(n) { it }, result.copyRowIndices())
            for (j in 0 until n) assertEquals((j % 7 + 1.0) * (j % 7 + 1.0), result.values[j])
        }
    }

    @Test
    fun `sparse syrk handles hypersparse rectangular adjacency`() {
        val columns = List(193) { j ->
            when (j) {
                2 -> listOf(0 to 2.0, 256 to -3.0)
                191 -> listOf(128 to 4.0)
                else -> emptyList()
            }
        }
        val a = SparseMatrix.ofColumns(257, 193, columns)

        for (lower in booleanArrayOf(false, true)) {
            val ordinary = ReferenceSparseBlas.syrk(a, transpose = false, lower)
            assertEquals(4, ordinary.nnz)
            assertEquals(4.0, ordinary[0, 0])
            assertEquals(16.0, ordinary[128, 128])
            assertEquals(9.0, ordinary[256, 256])
            assertEquals(-6.0, if (lower) ordinary[256, 0] else ordinary[0, 256])

            val transposed = ReferenceSparseBlas.syrk(a, transpose = true, lower)
            assertEquals(2, transposed.nnz)
            assertEquals(13.0, transposed[2, 2])
            assertEquals(16.0, transposed[191, 191])
        }
    }

    @Test
    fun `empty products retain requested shapes and scalar behavior`() {
        val a = SparseMatrix.ofColumns(2, 0, emptyList())
        val b = SparseMatrix.ofColumns(0, 3, List(3) { emptyList() })
        val sparse = ReferenceSparseBlas.gemm(2.0, a, false, b, false)
        assertEquals(2, sparse.rows)
        assertEquals(3, sparse.cols)
        assertEquals(0, sparse.nnz)

        val dense = DenseMatrix.wrap(2, 3, DoubleArray(6) { 4.0 })
        ReferenceSparseBlas.gemm(0.0, a, false, b, false, -0.5, dense)
        assertTrue(dense.data.all { it == -2.0 })

        val emptySymmetric = SparseMatrix.ofColumns(0, 0, emptyList())
        ReferenceSparseBlas.symv(1.0, emptySymmetric, DoubleArray(0), 0.0, DoubleArray(0))
        ReferenceSparseBlas.symm(
            1.0,
            emptySymmetric,
            DenseMatrix.zero(0, 3),
            0.0,
            DenseMatrix.zero(0, 3),
        )
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
        val zero = ReferenceSparseBlas.addScaled(-0.0, a, false, b)
        assertContentEquals(intArrayOf(0, 2, 3), zero.copyColumnPointers())
        assertEquals(5.0, zero[0, 0])
        assertEquals(7.0, zero[1, 0])
        assertEquals(-3.0, zero[1, 1])

        val aOnly = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to Double.NaN)))
        val empty = SparseMatrix.ofColumns(1, 1, listOf(emptyList()))
        val signedZero = ReferenceSparseBlas.addScaled(-0.0, aOnly, false, empty)
        assertEquals((-0.0).toBits(), signedZero.values.single().toBits())
    }
}
