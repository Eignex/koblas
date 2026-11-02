package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.PortableEngine
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.engineWithKernels
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparseKernelsTest {
    private val kernels
        get() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).distinct().map { it } +
            engineWithKernels(indexed = CustomIndexedDot())

    @Test
    fun `column dots agree with the scalar oracle across support lengths and offsets`() {
        for (count in listOf(0, 1, 7, 8, 9, 31, 64, 129, 512, 4096)) {
            val rows = count * 2 + 3
            val indices = intArrayOf(0, 1, 2) + IntArray(count) { 2 * it + 1 } + intArrayOf(rows - 1)
            val values = DoubleArray(indices.size) { (it % 17 - 8) * 0.125 }
            val matrix = SparseMatrix.wrap(
                rows,
                5,
                intArrayOf(0, 0, 3, 3 + count, 4 + count, 4 + count),
                indices,
                values,
            )
            val dense = DoubleArray(rows) { (it % 23 - 11) * 0.0625 }
            for (column in 0 until matrix.cols) {
                val start = matrix.colPointers[column]
                val length = matrix.colPointers[column + 1] - start
                val expected = BuiltinEngines.scalar.dot(indices, start, values, start, length, dense)

                for (selected in kernels) {
                    val actual = selected.dotColumn(matrix, column, dense)

                    assertDotAgreesWithReference(expected, actual, "${selected.name} count=$count column=$column")
                }
            }
        }
    }

    @Test
    fun `column dots evaluate stored zeros and skip absent nonfinite positions`() {
        for (stored in listOf(0.0, -0.0, 2.0, Double.POSITIVE_INFINITY, Double.NaN)) {
            val matrix = SparseMatrix.wrap(
                4,
                3,
                intArrayOf(0, 1, 2, 3),
                intArrayOf(0, 2, 3),
                doubleArrayOf(Double.NaN, stored, Double.NaN),
            )
            for (rhs in listOf(3.0, Double.POSITIVE_INFINITY, Double.NaN)) {
                val dense = doubleArrayOf(Double.NaN, Double.NEGATIVE_INFINITY, rhs, Double.NaN)
                val expected = BuiltinEngines.scalar.dot(
                    matrix.rowIndices,
                    1,
                    matrix.values,
                    1,
                    1,
                    dense,
                )

                for (selected in kernels) {
                    val actual = selected.dotColumn(matrix, 1, dense)

                    assertDotAgreesWithReference(expected, actual, "${selected.name} stored=$stored rhs=$rhs")
                }
            }
        }
    }

    @Test
    fun `column dots see coefficient updates without retaining a snapshot`() {
        val matrix = SparseMatrix.wrap(
            4,
            3,
            intArrayOf(0, 1, 3, 4),
            intArrayOf(3, 0, 2, 1),
            doubleArrayOf(8.0, 2.0, -3.0, 9.0),
        )
        val dense = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        for (selected in kernels) {
            for (coefficient in listOf(2.0, 5.0, -0.0)) {
                matrix.values[1] = coefficient
                val expected = BuiltinEngines.scalar.dot(matrix.rowIndices, 1, matrix.values, 1, 2, dense)

                val actual = selected.dotColumn(matrix, 1, dense)

                assertDotAgreesWithReference(expected, actual, "${selected.name} coefficient=$coefficient")
            }
        }
    }

    @Test
    fun `column dots allow the coefficient buffer as the dense operand without mutation`() {
        val matrix = SparseMatrix.wrap(
            4,
            3,
            intArrayOf(0, 1, 3, 4),
            intArrayOf(3, 0, 2, 1),
            doubleArrayOf(8.0, 2.0, -3.0, 9.0),
        )
        val before = matrix.values.copyOf()
        val expected = BuiltinEngines.scalar.dot(matrix.rowIndices, 1, before, 1, 2, before)

        for (selected in kernels) {
            val actual = selected.dotColumn(matrix, 1, matrix.values)

            assertDotAgreesWithReference(expected, actual, selected.name)
            assertContentEquals(before, matrix.values, selected.name)
        }
    }

    @Test
    fun `column dots accept empty columns with no rows`() {
        val matrix = SparseMatrix.wrap(0, 2, intArrayOf(0, 0, 0), IntArray(0), DoubleArray(0))
        val dense = DoubleArray(0)
        val expected = BuiltinEngines.scalar.dot(IntArray(0), 0, DoubleArray(0), 0, 0, dense)

        for (selected in kernels) {
            for (column in 0 until matrix.cols) {
                assertDotAgreesWithReference(expected, selected.dotColumn(matrix, column, dense), selected.name)
            }
        }
    }

    @Test
    fun `column dots reject invalid columns including a matrix with no columns`() {
        for (columns in listOf(0, 2)) {
            val matrix = SparseMatrix.wrap(3, columns, IntArray(columns + 1), IntArray(0), DoubleArray(0))
            for (selected in kernels) {
                for (column in listOf(Int.MIN_VALUE, -1, columns, Int.MAX_VALUE)) {
                    assertFailsWith<IndexOutOfBoundsException>(selected.name) {
                        selected.dotColumn(matrix, column, DoubleArray(matrix.rows))
                    }
                }
            }
        }
    }

    @Test
    fun `column dots reject dense shape mismatch even for empty support`() {
        val matrix = SparseMatrix.wrap(3, 1, intArrayOf(0, 0), IntArray(0), DoubleArray(0))
        for (selected in kernels) {
            for (length in listOf(0, 2, 4)) {
                assertFailsWith<IllegalArgumentException>(selected.name) {
                    selected.dotColumn(matrix, 0, DoubleArray(length))
                }
            }
        }
    }

    @Test
    fun `custom column dot defaults dispatch to the raw overload`() {
        val backend = CustomIndexedDot()
        val selected = engineWithKernels(indexed = backend)
        val matrix = SparseMatrix.wrap(
            5,
            3,
            intArrayOf(0, 1, 3, 3),
            intArrayOf(4, 0, 3),
            doubleArrayOf(9.0, 2.0, -3.0),
        )
        val dense = DoubleArray(matrix.rows)

        selected.dotColumn(matrix, 1, dense)
        selected.dotColumn(matrix, 2, dense)

        assertEquals(listOf(1 to 2, 3 to 0), backend.rawCalls)
    }

    private fun assertDotAgreesWithReference(expected: Double, actual: Double, context: String) {
        if (expected.isFinite() && expected != 0.0) {
            assertTrue(abs(actual - expected) <= 1e-12 * (1.0 + abs(expected)), context)
        } else {
            assertEquals(expected.toBits(), actual.toBits(), context)
        }
    }

    private class CustomIndexedDot : KoblasEngine() {
        override val name = "custom-indexed"
        override val implementations = setOf(EngineImplementation.IndexedDot)
        val rawCalls = mutableListOf<Pair<Int, Int>>()
        override fun dotDense(
            indices: IntArray,
            indexOffset: Int,
            values: DoubleArray,
            valueOffset: Int,
            count: Int,
            dense: DoubleArray,
        ): Double {
            rawCalls += indexOffset to count
            return PortableEngine.dotDense(indices, indexOffset, values, valueOffset, count, dense)
        }
    }
}
