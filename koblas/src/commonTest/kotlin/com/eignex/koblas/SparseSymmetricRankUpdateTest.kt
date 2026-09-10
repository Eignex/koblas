package com.eignex.koblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.ReferenceBlas
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparseSymmetricRankUpdateTest {

    @Test
    fun `sparse syr agrees with the dense reference and preserves CSC`() {
        val source = SparseMatrix.ofColumns(
            4,
            4,
            listOf(
                listOf(0 to 1.0, 3 to 9.0),
                listOf(0 to -3.0, 1 to 0.0),
                listOf(2 to 5.0),
                listOf(1 to 7.0),
            ),
        )
        val x = SparseVector.of(4, intArrayOf(3, 0, 3, 2), doubleArrayOf(2.0, -1.0, 1.0, 4.0))

        for (lower in booleanArrayOf(true, false)) {
            val expected = source.denseCopy()
            ReferenceBlas.syr(0.5, x, expected, lower)

            val actual = source.syr(0.5, x, lower)

            assertMatrixEquals(expected, actual, "lower=$lower")
            assertCanonical(actual)
            assertEquals(0.0, actual[1, 1], "lower=$lower preserves explicit zero")
        }
    }

    @Test
    fun `sparse syr2 accepts dense and sparse vectors and agrees with the dense reference`() {
        val source = SparseMatrix.ofColumns(
            4,
            4,
            listOf(listOf(0 to 1.0), listOf(3 to -2.0), listOf(1 to 3.0), emptyList()),
        )
        val x = DenseVector.of(doubleArrayOf(2.0, 0.0, -1.0, 3.0))
        val y = SparseVector.of(4, intArrayOf(3, 0, 3, 1), doubleArrayOf(-1.0, 4.0, 2.0, 5.0))

        for (lower in booleanArrayOf(true, false)) {
            val expected = source.denseCopy()
            ReferenceBlas.syr2(-0.75, x, y, expected, lower)

            val actual = source.syr2(-0.75, x, y, lower)

            assertMatrixEquals(expected, actual, "lower=$lower")
            assertCanonical(actual)
        }
    }

    @Test
    fun `sparse rank updates retain cancelled fill and do not alias their inputs`() {
        val source = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 0.0), emptyList()))
        val x = DenseVector.wrap(doubleArrayOf(1.0, 1.0))
        val y = SparseVector.of(2, intArrayOf(0, 1), doubleArrayOf(1.0, -1.0))

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
        val source = SparseMatrix.ofColumns(3, 3, listOf(emptyList(), emptyList(), emptyList()))
        val x = SparseVector.of(3, intArrayOf(0), doubleArrayOf(Double.POSITIVE_INFINITY))
        val y = SparseVector.of(3, intArrayOf(2), doubleArrayOf(2.0))

        for (lower in booleanArrayOf(true, false)) {
            val expected = source.denseCopy()
            ReferenceBlas.syr2(1.0, x, y, expected, lower)

            val actual = source.syr2(1.0, x, y, lower)

            assertMatrixEquals(expected, actual, "lower=$lower")
        }
    }

    @Test
    fun `sparse syr excludes an explicitly stored zero in a sparse operand from its fill support`() {
        val source = SparseMatrix.ofColumns(3, 3, listOf(emptyList(), emptyList(), emptyList()))
        val x = SparseVector.of(3, intArrayOf(0, 1, 2), doubleArrayOf(1.0, 0.0, 2.0))

        for (lower in booleanArrayOf(true, false)) {
            val expected = source.denseCopy()
            ReferenceBlas.syr(1.0, x, expected, lower)

            val actual = source.syr(1.0, x, lower)

            assertMatrixEquals(expected, actual, "lower=$lower")
            assertCanonical(actual)
            assertEquals(3, actual.nnz, "lower=$lower a stored zero must not seed spurious fill")
        }
    }

    @Test
    fun `sparse syr2 excludes an explicitly stored zero in a sparse operand from its fill support`() {
        val source = SparseMatrix.ofColumns(3, 3, listOf(emptyList(), emptyList(), emptyList()))
        val x = SparseVector.of(3, intArrayOf(0, 1, 2), doubleArrayOf(1.0, 0.0, 2.0))
        val y = SparseVector.of(3, intArrayOf(0, 2), doubleArrayOf(-3.0, 4.0))

        for (lower in booleanArrayOf(true, false)) {
            val expected = source.denseCopy()
            ReferenceBlas.syr2(1.0, x, y, expected, lower)

            val actual = source.syr2(1.0, x, y, lower)

            assertMatrixEquals(expected, actual, "lower=$lower")
            assertCanonical(actual)
            assertEquals(3, actual.nnz, "lower=$lower a stored zero must not seed spurious fill")
        }
    }

    @Test
    fun `sparse syr keeps a positive sign on an underflowing pure fill entry`() {
        val alpha = 1e-200
        val x = DenseVector.of(doubleArrayOf(-1e-200, 1.0))
        val source = SparseMatrix.ofColumns(2, 2, listOf(emptyList(), emptyList()))

        val actual = source.syr(alpha, x, lower = true)

        val expected = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 0.0, 1 to 0.0), listOf(1 to alpha)))
        assertEquals(expected, actual, "underflowing fill must land on +0.0, not -0.0")
    }

    @Test
    fun `sparse rank updates support empty shapes and reject incompatible operands`() {
        val empty = SparseMatrix.ofColumns(0, 0, emptyList())

        assertEquals(empty, empty.syr(1.0, DenseVector.zero(0)))
        assertEquals(
            empty,
            empty.syr2(
                1.0,
                DenseVector.zero(0),
                SparseVector.of(0, IntArray(0), DoubleArray(0)),
            ),
        )

        val rectangular = SparseMatrix.ofColumns(
            2,
            3,
            listOf(emptyList(), emptyList(), emptyList()),
        )
        assertFailsWith<DimensionMismatch> { rectangular.syr(1.0, DenseVector.zero(2)) }
        assertFailsWith<DimensionMismatch> {
            SparseMatrix.ofColumns(2, 2, listOf(emptyList(), emptyList())).syr(1.0, DenseVector.zero(3))
        }
        assertFailsWith<DimensionMismatch> {
            SparseMatrix.ofColumns(2, 2, listOf(emptyList(), emptyList())).syr2(
                1.0,
                DenseVector.zero(2),
                DenseVector.zero(3),
            )
        }
    }

    private fun assertMatrixEquals(expected: DenseMatrix, actual: SparseMatrix, context: String) {
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

    private fun assertCanonical(matrix: SparseMatrix) {
        val pointers = matrix.copyColumnPointers()
        val rows = matrix.copyRowIndices()
        for (j in 0 until matrix.cols) {
            val columnRows = rows.copyOfRange(pointers[j], pointers[j + 1])
            assertContentEquals(columnRows.sortedArray(), columnRows, "column $j rows ascend")
        }
    }
}
