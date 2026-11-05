package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.BuiltinEngines
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame

class MatrixOpsTest {
    @Test
    fun `total sums agree with the scalar oracle across storage types`() {
        for (a in reductionMatrices()) {
            val entries = DoubleArray(a.rows * a.cols) { a[it % a.rows, it / a.rows] }
            val expected = BuiltinEngines.scalar.sum(entries, 0, entries.size)

            val actual = a.sum()

            assertSumAgreesWithReference(expected, actual)
        }
    }

    @Test
    fun `axis sums reduce the selected dimension across storage types`() {
        for (a in reductionMatrices()) {
            for (axis in intArrayOf(0, 1)) {
                val expected = DoubleArray(if (axis == 0) a.cols else a.rows) { index ->
                    val entries = if (axis == 0) {
                        DoubleArray(a.rows) { a[it, index] }
                    } else {
                        DoubleArray(a.cols) { a[index, it] }
                    }
                    BuiltinEngines.scalar.sum(entries, 0, entries.size)
                }

                val actual = a.sum(axis)

                assertVectorAgreesWithReference(expected, actual, DoubleArray(expected.size))
            }
        }
    }

    @Test
    fun `axis sums return independent storage and leave the source unchanged`() {
        for (a in reductionMatrices()) {
            val before = a.toArray()
            for (axis in intArrayOf(0, 1)) {
                val first = a.sum(axis)
                val second = a.sum(axis)

                first[0] = 123.0

                assertNotSame(first.values, second.values)
                for (i in 0 until a.rows) for (j in 0 until a.cols) assertEquals(before[i][j], a[i, j])
                assertEquals(a.sum(axis)[0], second[0])
            }
        }
    }

    @Test
    fun `empty dimensions produce sums with the surviving shape`() {
        for ((rows, cols) in listOf(0 to 0, 0 to 3, 4 to 0)) {
            val dense = DenseMatrix.zero(rows, cols)
            val sparse = SparseMatrix.ofColumns(rows, cols, List(cols) { emptyList<Pair<Int, Double>>() })
            val view = paddedRegion(rows, cols)
            for (a in listOf(dense, sparse, view, ForeignMatrix(dense))) {
                assertEquals(0.0, a.sum())
                assertContentEquals(DoubleArray(cols), a.sum(0).toDoubleArray())
                assertContentEquals(DoubleArray(rows), a.sum(1).toDoubleArray())
            }
        }
    }

    @Test
    fun `invalid sum axes are rejected even for empty matrices`() {
        for (a in reductionMatrices() + DenseMatrix.zero(0, 0)) {
            for (axis in intArrayOf(Int.MIN_VALUE, -1, 2, Int.MAX_VALUE)) {
                assertFailsWith<IllegalArgumentException> { a.sum(axis) }
            }
        }
    }

    @Test
    fun `sparse sums depend only on stored entries for large shapes`() {
        val a = SparseMatrix.ofColumns(Int.MAX_VALUE, 2, listOf(listOf(0 to 1.25), listOf(Int.MAX_VALUE - 1 to -0.5)))
        val values = doubleArrayOf(1.25, -0.5)
        val expected = BuiltinEngines.scalar.sum(values, 0, values.size)

        assertSumAgreesWithReference(expected, a.sum())
        assertVectorAgreesWithReference(values, a.sum(0), DoubleArray(2))
    }

    private fun reductionMatrices(): List<Matrix> {
        val sparse = SparseMatrix.ofColumns(
            3,
            4,
            listOf(listOf(0 to 2.5, 2 to -1.0), listOf(0 to 0.0, 1 to 4.0), emptyList(), listOf(1 to -2.0, 2 to 0.75)),
        )
        val dense = sparse.denseCopy()
        val view = paddedRegion(3, 4)
        for (j in 0 until view.cols) for (i in 0 until view.rows) view[i, j] = dense[i, j]
        return listOf(dense, view, sparse, ForeignMatrix(dense))
    }

    private class ForeignMatrix(private val source: Matrix) : Matrix {
        override val rows: Int get() = source.rows
        override val cols: Int get() = source.cols
        override fun get(i: Int, j: Int): Double = source[i, j]
        override fun toArray(): Array<DoubleArray> = source.toArray()
    }

    private fun assertSumAgreesWithReference(expected: Double, actual: Double) {
        assertClose(expected, actual, "matrix sum")
    }

    @Test
    fun `matrix vector product snapshots overlapping strided operands`() {
        for (transpose in booleanArrayOf(false, true)) {
            for (stride in intArrayOf(-2, 2)) {
                for (beta in doubleArrayOf(0.0, -0.25)) {
                    val a = paddedRegion(3, 4)
                    val x = StridedVector(a.values, a.offset + 8, if (transpose) 3 else 4, stride)
                    val y = StridedVector(a.values, a.offset + 8, if (transpose) 4 else 3, -stride)
                    val before = a.values.copyOf()
                    val expected = y.toDoubleArray()
                    ReferenceBlas.gemv(-0.75, contiguousCopy(a), x.toDoubleArray(), beta, expected, transpose)

                    a.gemvInto(x, y, alpha = -0.75, beta = beta, transpose = transpose, scratch = Scratch())

                    assertVectorAgreesWithReference(expected, y, before)
                }
            }
        }
    }

    @Test
    fun `matrix vector product defaults replace the destination`() {
        val a = paddedRegion(3, 4)
        val x = DenseVector.of(doubleArrayOf(0.25, -1.0, 2.0, 0.5))
        val y = DenseVector.wrap(DoubleArray(3) { Double.NaN })
        val expected = DoubleArray(3)
        ReferenceBlas.gemv(1.0, a, x.toDoubleArray(), 0.0, expected)

        a.gemvInto(x, y)

        assertVectorAgreesWithReference(expected, y, DoubleArray(3))
    }

    @Test
    fun `matrix product preserves overlapping rectangles across transpose flags`() {
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val parent = paddedRegion(6, 6)
                val a = parent.view(rows = if (transposeA) 4 else 3, cols = if (transposeA) 3 else 4)
                val b = parent.view(1, 1, if (transposeB) 2 else 4, if (transposeB) 4 else 2)
                val destination = parent.view(rows = 3, cols = 2)
                val before = parent.values.copyOf()
                val expected = contiguousCopy(destination)
                ReferenceBlas.gemm(-0.75, contiguousCopy(a), transposeA, contiguousCopy(b), transposeB, 0.25, expected)

                a.gemmInto(b, destination, -0.75, 0.25, transposeA, transposeB, Scratch())

                assertRegionAgreesWithReference(expected, destination, before)
            }
        }
    }

    @Test
    fun `matrix product defaults replace the destination`() {
        val a = paddedRegion(3, 4)
        val b = paddedRegion(4, 2)
        val destination = paddedRegion(3, 2)
        for (j in 0 until destination.cols) for (i in 0 until destination.rows) destination[i, j] = Double.NaN
        val before = destination.values.copyOf()
        val expected = DenseMatrix.zero(3, 2)
        ReferenceBlas.gemm(1.0, a, false, b, false, 0.0, expected)

        a.gemmInto(b, destination)

        assertRegionAgreesWithReference(expected, destination, before)
    }

    @Test
    fun `general rank update snapshots both strided vectors sharing matrix storage`() {
        for (alpha in doubleArrayOf(1.0, -0.75)) {
            val a = paddedRegion(3, 4)
            val x = StridedVector(a.values, a.offset, a.rows, 1)
            val y = StridedVector(a.values, a.offset + 6, a.cols, -2)
            val before = a.values.copyOf()
            val expected = contiguousCopy(a)
            ReferenceBlas.ger(alpha, x.toDoubleArray(), y.toDoubleArray(), expected)

            if (alpha == 1.0) {
                a.ger(x, y)
            } else {
                a.ger(x, y, alpha = alpha, scratch = Scratch())
            }

            assertRegionAgreesWithReference(expected, a, before)
        }
    }

    @Test
    fun `symmetric rank update snapshots a vector sharing matrix storage`() {
        for (lower in booleanArrayOf(false, true)) {
            for (stride in intArrayOf(-1, 1)) {
                val a = paddedRegion(4, 4)
                val x = StridedVector(a.values, a.offset + if (stride < 0) 3 else 0, 4, stride)
                val before = a.values.copyOf()
                val expected = contiguousCopy(a)
                ReferenceBlas.syr(-0.75, DenseVector.of(x.toDoubleArray()), expected, lower)

                a.syr(x, alpha = -0.75, lower = lower, scratch = Scratch())

                assertRegionAgreesWithReference(expected, a, before)
            }
        }
    }

    @Test
    fun `triangular vector solve preserves aliased views across flags`() {
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unitDiag in booleanArrayOf(false, true)) {
                    for (stride in intArrayOf(-1, 1)) {
                        val a = paddedRegion(4, 4)
                        val x = StridedVector(a.values, a.offset + if (stride < 0) 3 else 0, 4, stride)
                        val before = a.values.copyOf()
                        val expected = x.toDoubleArray()
                        ReferenceBlas.trsv(contiguousCopy(a), expected, lower, transpose, unitDiag)

                        a.trsv(x, lower, transpose, unitDiag, Scratch())

                        assertVectorAgreesWithReference(expected, x, before)
                    }
                }
            }
        }
    }

    @Test
    fun `triangular matrix solve snapshots an overlapping factor across flags`() {
        for (right in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val a = paddedRegion(4, 4)
                        val destination = a.view(rows = if (right) 3 else 4, cols = if (right) 4 else 3)
                        val before = a.values.copyOf()
                        val expected = contiguousCopy(destination)
                        ReferenceBlas.trsm(contiguousCopy(a), expected, lower, transpose, unitDiag, right, -0.75)

                        a.trsm(destination, lower, transpose, unitDiag, right, -0.75, Scratch())

                        assertRegionAgreesWithReference(expected, destination, before)
                    }
                }
            }
        }
    }

    @Test
    fun `vector sizes are validated before writing even at zero alpha`() {
        val a = paddedRegion(3, 3)
        val before = a.values.copyOf()
        val valid = DenseVector.zero(3)
        for (size in intArrayOf(2, 4)) {
            // Both sizes have enough backing storage for a raw window of three entries.
            val wrong = StridedVector(a.values, a.offset, size)
            for (transpose in booleanArrayOf(false, true)) {
                assertFailsWith<DimensionMismatch> { a.gemvInto(wrong, valid, alpha = 0.0, transpose = transpose) }
                assertFailsWith<DimensionMismatch> { a.gemvInto(valid, wrong, alpha = 0.0, transpose = transpose) }
            }
            assertFailsWith<DimensionMismatch> { a.ger(wrong, valid, alpha = 0.0) }
            assertFailsWith<DimensionMismatch> { a.ger(valid, wrong, alpha = 0.0) }
            assertFailsWith<DimensionMismatch> { a.syr(wrong, alpha = 0.0) }
            assertFailsWith<DimensionMismatch> { a.trsv(wrong, lower = true) }
        }

        assertContentEquals(before, a.values)
        assertContentEquals(DoubleArray(3), valid.values)
    }

    private fun assertVectorAgreesWithReference(expected: DoubleArray, actual: DenseVector, before: DoubleArray) {
        val backing = before.copyOf()
        for (i in expected.indices) backing[actual.offset + i * actual.stride] = expected[i]
        assertClose(backing, actual.values, "vector backing buffer")
    }
}
