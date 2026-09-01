package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparseSymmetricRankUpdateTest {

    @Test
    fun `sparse syr agrees with the dense reference and preserves CSC`() {
        val source = F64SparseMatrix.ofColumns(
            4,
            4,
            listOf(
                listOf(0 to 1.0, 3 to 9.0),
                listOf(0 to -3.0, 1 to 0.0),
                listOf(2 to 5.0),
                listOf(1 to 7.0),
            ),
        )
        val x = F64SparseVector.of(4, intArrayOf(3, 0, 3, 2), doubleArrayOf(2.0, -1.0, 1.0, 4.0))

        for (lower in booleanArrayOf(true, false)) {
            val expected = denseCopy(source)
            F64ReferenceLinearAlgebra.syr(0.5, x, expected, lower)

            val actual = source.syr(0.5, x, lower)

            assertMatrixEquals(expected, actual, "lower=$lower")
            assertCanonical(actual)
            assertEquals(0.0, actual[1, 1], "lower=$lower preserves explicit zero")
        }
    }

    @Test
    fun `sparse syr2 accepts dense and sparse vectors and agrees with the dense reference`() {
        val source = F64SparseMatrix.ofColumns(
            4,
            4,
            listOf(listOf(0 to 1.0), listOf(3 to -2.0), listOf(1 to 3.0), emptyList()),
        )
        val x = F64DenseVector.of(doubleArrayOf(2.0, 0.0, -1.0, 3.0))
        val y = F64SparseVector.of(4, intArrayOf(3, 0, 3, 1), doubleArrayOf(-1.0, 4.0, 2.0, 5.0))

        for (lower in booleanArrayOf(true, false)) {
            val expected = denseCopy(source)
            F64ReferenceLinearAlgebra.syr2(-0.75, x, y, expected, lower)

            val actual = source.syr2(-0.75, x, y, lower)

            assertMatrixEquals(expected, actual, "lower=$lower")
            assertCanonical(actual)
        }
    }

    @Test
    fun `sparse rank updates retain cancelled fill and do not alias their inputs`() {
        val source = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 0.0), emptyList()))
        val x = F64DenseVector.wrap(doubleArrayOf(1.0, 1.0))
        val y = F64SparseVector.of(2, intArrayOf(0, 1), doubleArrayOf(1.0, -1.0))

        val rankOne = source.syr(0.0, x)
        val rankTwo = source.syr2(1.0, x, y)
        source.values[0] = 8.0
        x[0] = 9.0
        y.values[0] = 7.0

        assertEquals(0.0, rankOne[0, 0])
        assertEquals(2.0, rankTwo[0, 0])
        assertEquals(0.0, rankTwo[1, 0])
        assertEquals(-2.0, rankTwo[1, 1])
        assertEquals(3, rankTwo.nnz, "cancelled fill remains explicit")
        assertCanonical(rankTwo)
    }

    @Test
    fun `sparse rank updates match dense IEEE arithmetic for implicit zeros`() {
        val source = F64SparseMatrix.ofColumns(3, 3, listOf(emptyList(), emptyList(), emptyList()))
        val x = F64SparseVector.of(3, intArrayOf(0), doubleArrayOf(Double.POSITIVE_INFINITY))
        val y = F64SparseVector.of(3, intArrayOf(2), doubleArrayOf(2.0))

        for (lower in booleanArrayOf(true, false)) {
            val expected = denseCopy(source)
            F64ReferenceLinearAlgebra.syr2(1.0, x, y, expected, lower)

            val actual = source.syr2(1.0, x, y, lower)

            assertMatrixEquals(expected, actual, "lower=$lower")
        }
    }

    @Test
    fun `sparse rank updates support empty shapes and reject incompatible operands`() {
        val empty = F64SparseMatrix.ofColumns(0, 0, emptyList())

        assertEquals(empty, empty.syr(1.0, F64DenseVector.zero(0)))
        assertEquals(
            empty,
            empty.syr2(
                1.0,
                F64DenseVector.zero(0),
                F64SparseVector.of(0, IntArray(0), DoubleArray(0)),
            ),
        )

        val rectangular = F64SparseMatrix.ofColumns(
            2,
            3,
            listOf(emptyList(), emptyList(), emptyList()),
        )
        assertFailsWith<DimensionMismatch> { rectangular.syr(1.0, F64DenseVector.zero(2)) }
        assertFailsWith<DimensionMismatch> {
            F64SparseMatrix.ofColumns(2, 2, listOf(emptyList(), emptyList())).syr2(
                1.0,
                F64DenseVector.zero(2),
                F64DenseVector.zero(3),
            )
        }
    }

    private fun denseCopy(source: F64SparseMatrix): F64DenseMatrix = F64DenseMatrix(
        source.rows,
        source.cols,
    ).also { out ->
        for (j in 0 until source.cols) source.forEachInColumn(j) { i, value -> out[i, j] = value }
    }

    private fun assertMatrixEquals(expected: F64DenseMatrix, actual: F64SparseMatrix, context: String) {
        for (j in 0 until expected.cols) {
            for (i in 0 until expected.rows) {
                val expectedValue = expected[i, j]
                val actualValue = actual[i, j]
                assertTrue(
                    expectedValue == actualValue || (expectedValue.isNaN() && actualValue.isNaN()),
                    "$context ($i, $j): expected $expectedValue, got $actualValue",
                )
            }
        }
    }

    private fun assertCanonical(matrix: F64SparseMatrix) {
        val pointers = matrix.copyColumnPointers()
        val rows = matrix.copyRowIndices()
        for (j in 0 until matrix.cols) {
            val columnRows = rows.copyOfRange(pointers[j], pointers[j + 1])
            assertContentEquals(columnRows.sortedArray(), columnRows, "column $j rows ascend")
        }
    }
}
