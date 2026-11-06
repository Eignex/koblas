package com.eignex.koblas

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.kernels.*
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotSame

class MatrixOpsTest {
    @Test
    fun `stored traversal visits dense views and custom storage in column order`() {
        val dense = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 0.0, -2.0), doubleArrayOf(3.0, 4.0, 5.0)))
        val parent = DenseMatrix.wrap(5, 6, DoubleArray(30) { Double.NaN })
        val view = parent.view(1, 2, 2, 3)
        for (j in 0 until dense.cols) for (i in 0 until dense.rows) view[i, j] = dense[i, j]
        val expected = listOf(
            Triple(0, 0, 1.0), Triple(1, 0, 3.0),
            Triple(0, 1, 0.0), Triple(1, 1, 4.0),
            Triple(0, 2, -2.0), Triple(1, 2, 5.0),
        )
        for (source in listOf<Matrix>(dense, view, ForeignMatrix(dense), ForeignMatrix(view))) {
            val actual = mutableListOf<Triple<Int, Int, Double>>()

            source.forEachStored { row, col, value -> actual += Triple(row, col, value) }

            assertEquals(expected, actual)
        }
    }

    @Test
    fun `stored traversal preserves sparse zeros and skips absent positions in large shapes`() {
        val source: Matrix = SparseMatrix.ofColumns(
            Int.MAX_VALUE,
            4,
            listOf(listOf(1 to 2.0, Int.MAX_VALUE - 1 to -0.0), emptyList(), listOf(0 to 0.0), emptyList()),
        )
        val actual = mutableListOf<Triple<Int, Int, Double>>()

        source.forEachStored { row, col, value -> actual += Triple(row, col, value) }

        assertEquals(listOf(Triple(1, 0, 2.0), Triple(Int.MAX_VALUE - 1, 0, -0.0), Triple(0, 2, 0.0)), actual)
    }

    @Test
    fun `stored traversal invokes no callbacks for empty matrix dimensions`() {
        val sources: List<Matrix> = listOf(
            DenseMatrix.zero(0, Int.MAX_VALUE),
            DenseMatrix.zero(Int.MAX_VALUE, 0),
            SparseMatrix.ofColumns(0, 3, List(3) { emptyList() }),
            SparseMatrix.ofColumns(Int.MAX_VALUE, 0, emptyList()),
        )
        for (source in sources + sources.map { ForeignMatrix(it) }) {
            source.forEachStored { _, _, _ -> error("Unexpected stored entry") }
        }
    }

    @Test
    fun `empty matrix copies and transposes do not traverse the column domain`() {
        val source: Matrix = DenseMatrix.zero(0, Int.MAX_VALUE)
        val destination: Matrix = DenseMatrix.zero(0, Int.MAX_VALUE)

        val result = source.copy()
        val transpose = source.transpose()
        destination.copyFrom(source)

        assertEquals(0, result.rows)
        assertEquals(Int.MAX_VALUE, result.cols)
        assertEquals(Int.MAX_VALUE, transpose.rows)
        assertEquals(0, transpose.cols)
    }

    @Test
    fun `sparse matrix copies preserve large shapes and stored zeros without densifying`() {
        val source: Matrix = SparseMatrix.ofColumns(
            Int.MAX_VALUE,
            2,
            listOf(listOf(0 to 0.0), listOf(Int.MAX_VALUE - 1 to -2.0)),
        )
        val destination: Matrix = SparseMatrix.ofColumns(Int.MAX_VALUE, 2, listOf(emptyList(), emptyList()))

        val result = source.copy()
        destination.copyFrom(source)

        assertEquals(source, result)
        assertEquals(source, destination)
        assertNotSame(assertIs<SparseMatrix>(source).values, assertIs<SparseMatrix>(result).values)
        assertNotSame(source.values, assertIs<SparseMatrix>(destination).values)
    }

    @Test
    fun `copies own their storage across matrix implementations`() {
        for (source in reductionMatrices()) {
            val result = source.copy()

            for (j in 0 until source.cols) for (i in 0 until source.rows) assertEquals(source[i, j], result[i, j])
            if (source is SparseMatrix) {
                val sparseCopy = assertIs<SparseMatrix>(result)
                assertContentEquals(source.colPointers, sparseCopy.colPointers)
                assertContentEquals(source.rowIndices, sparseCopy.rowIndices)
                assertNotSame(source.colPointers, sparseCopy.colPointers)
                assertNotSame(source.rowIndices, sparseCopy.rowIndices)
                assertNotSame(source.values, sparseCopy.values)
            }
            val before = source[0, 0]
            result[0, 0] = 99.0
            assertEquals(before, source[0, 0])
        }
    }

    @Test
    fun `copy from supports every matrix storage without sharing arrays`() {
        for (source in reductionMatrices()) {
            for (destination in reductionMatrices(3.0)) {
                destination.copyFrom(source)

                for (j in 0 until source.cols) {
                    for (i in 0 until source.rows) assertEquals(source[i, j], destination[i, j])
                }
                if (source is SparseMatrix && destination is SparseMatrix) {
                    assertContentEquals(source.rowIndices, destination.rowIndices)
                    assertNotSame(source.values, destination.values)
                }
            }
        }
    }

    @Test
    fun `copy from snapshots overlapping matrix views and custom storage`() {
        for (customSource in booleanArrayOf(false, true)) {
            for (customDestination in booleanArrayOf(false, true)) {
                val parent = paddedRegion(5, 5)
                val source = parent.view(0, 0, 3, 2)
                val destination = parent.view(1, 1, 3, 2)
                val expected = contiguousCopy(source)
                val before = parent.values.copyOf()
                val input: Matrix = if (customSource) ForeignMatrix(source) else source
                val output: Matrix = if (customDestination) ForeignMatrix(destination) else destination

                output.copyFrom(input)

                assertRegionAgreesWithReference(expected, destination, before)
            }
        }
    }

    @Test
    fun `copy from validates the whole shape before writing`() {
        for (destination in reductionMatrices()) {
            val before = destination.toArray()

            assertFailsWith<DimensionMismatch> { destination.copyFrom(DenseMatrix.zero(4, 3)) }

            for (i in 0 until destination.rows) assertContentEquals(before[i], destination.toArray()[i])
        }
    }

    @Test
    fun `transpose accepts views and custom matrices through the base contract`() {
        for (source in reductionMatrices()) {
            val result = source.transpose()

            assertEquals(source.cols, result.rows)
            assertEquals(source.rows, result.cols)
            for (j in 0 until source.cols) for (i in 0 until source.rows) assertEquals(source[i, j], result[j, i])
            val before = source[0, 0]
            result[0, 0] = 123.0
            assertEquals(before, source[0, 0])
        }
    }

    @Test
    fun `matrix norms agree with the scalar oracle across storage types`() {
        for (a in reductionMatrices()) {
            for (type in listOf(Norm.One, Norm.Infinity, Norm.Frobenius, Norm.MaxAbs)) {
                val expected = referenceMatrixNorm(a, type)

                val actual = a.norm(type)

                assertNormAgreesWithReference(expected, actual)
            }
        }
    }

    @Test
    fun `matrix norm defaults to rescaled Frobenius across storage types`() {
        for (scale in doubleArrayOf(1e-200, 1.0, 1e200)) {
            for (a in reductionMatrices(scale)) {
                val expected = referenceMatrixNorm(a, Norm.Frobenius)

                val actual = a.norm()

                assertNormAgreesWithReference(expected / scale, actual / scale)
            }
        }
    }

    @Test
    fun `supported matrix norms of empty and zero matrices are zero`() {
        for ((rows, cols) in listOf(0 to 0, 0 to 3, 4 to 0, 3 to 4)) {
            val dense = DenseMatrix.zero(rows, cols)
            val sparse = SparseMatrix.ofColumns(rows, cols, List(cols) { emptyList<Pair<Int, Double>>() })
            val view = paddedRegion(rows, cols)
            for (j in 0 until cols) for (i in 0 until rows) view[i, j] = 0.0
            for (a in listOf(dense, sparse, view, ForeignMatrix(dense))) {
                for (type in listOf(Norm.One, Norm.Infinity, Norm.Frobenius, Norm.MaxAbs)) {
                    assertEquals(0.0, a.norm(type))
                }
            }
        }
    }

    @Test
    fun `spectral matrix norm is rejected before reading the matrix`() {
        val a = object : Matrix {
            override val rows: Int get() = error("Read row count")
            override val cols: Int get() = error("Read column count")
            override fun get(i: Int, j: Int): Double = error("Read entry")
            override fun set(i: Int, j: Int, v: Double) = error("Unexpected mutation")
            override fun toArray(): Array<DoubleArray> = error("Materialized matrix")
        }

        assertFailsWith<IllegalArgumentException> { a.norm(Norm.Two) }
        assertFailsWith<IllegalArgumentException> { DenseMatrix.zero(0, 0).norm(Norm.Two) }
    }

    @Test
    fun `sparse matrix norms scale with stored entries for large row domains`() {
        val a = SparseMatrix.ofColumns(
            Int.MAX_VALUE,
            2,
            listOf(listOf(0 to 2.0), listOf(0 to -3.0, Int.MAX_VALUE - 1 to 4.0)),
        )
        val compact = DenseMatrix.ofRows(arrayOf(doubleArrayOf(2.0, -3.0), doubleArrayOf(0.0, 4.0)))
        for (type in listOf(Norm.One, Norm.Infinity, Norm.Frobenius, Norm.MaxAbs)) {
            val expected = referenceMatrixNorm(compact, type)

            val actual = a.norm(type)

            assertNormAgreesWithReference(expected, actual)
        }
    }

    @Test
    fun `matrix norms leave the whole backing buffer unchanged`() {
        val a = paddedRegion(3, 4)
        val before = a.values.copyOf()

        for (type in listOf(Norm.One, Norm.Infinity, Norm.Frobenius, Norm.MaxAbs)) a.norm(type)

        assertContentEquals(before, a.values)
    }

    private fun referenceMatrixNorm(a: Matrix, type: Norm): Double {
        if (type == Norm.One || type == Norm.Infinity) {
            val rows = type == Norm.Infinity
            val sums = DoubleArray(if (rows) a.rows else a.cols) { index ->
                val entries = if (rows) DoubleArray(a.cols) { a[index, it] } else DoubleArray(a.rows) { a[it, index] }
                KoblasEngine.scalar.asum(entries, 0, entries.size)
            }
            return sums.maxOrNull() ?: 0.0
        }
        val entries = DoubleArray(a.rows * a.cols) { a[it % a.rows, it / a.rows] }
        if (type == Norm.Frobenius) return KoblasEngine.scalar.nrm2(entries, 0, entries.size)
        val index = KoblasEngine.scalar.iamax(entries, 0, entries.size)
        return if (index < 0) 0.0 else abs(entries[index])
    }

    private fun assertNormAgreesWithReference(expected: Double, actual: Double) {
        assertClose(expected, actual, "matrix norm")
    }

    @Test
    fun `dense transpose round trips and maps entries`() {
        val a = DenseMatrix.ofRows(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
            ),
        )
        val t = a.transpose()
        assertEquals(3, t.rows)
        assertEquals(2, t.cols)
        for (i in 0 until a.rows) for (j in 0 until a.cols) assertEquals(a[i, j], t[j, i])
        assertEquals(a, t.transpose())
        assertEquals(DenseMatrix(0, 0), DenseMatrix(0, 0).transpose())
        assertEquals(0, DenseMatrix(0, 5).transpose().cols)
        assertEquals(5, DenseMatrix(0, 5).transpose().rows)
    }

    @Test
    fun `sparse transpose round trips and preserves stored zeros`() {
        val a = SparseMatrix.ofColumns(
            3,
            2,
            listOf(
                listOf(0 to 4.0, 1 to 0.0, 2 to -1.0),
                listOf(1 to 7.0),
            ),
        )
        val t = assertIs<SparseMatrix>(a.transpose())
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(a.nnz, t.nnz, "an explicitly stored zero was dropped")
        for (i in 0 until a.rows) {
            for (j in 0 until a.cols) assertEquals(a[i, j], t[j, i], "transpose at [$i,$j]")
        }
        assertEquals(a, t.transpose(), "transpose twice is not the original")
    }

    @Test
    fun `sparse transpose handles degenerate shapes`() {
        val empty = SparseMatrix.ofColumns(0, 0, emptyList())
        assertEquals(empty, empty.transpose().transpose())
        val noEntries = SparseMatrix.ofColumns(3, 2, listOf(emptyList(), emptyList()))
        val t = assertIs<SparseMatrix>(noEntries.transpose())
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(0, t.nnz)
    }

    @Test
    fun `total sums agree with the scalar oracle across storage types`() {
        for (a in reductionMatrices()) {
            val entries = DoubleArray(a.rows * a.cols) { a[it % a.rows, it / a.rows] }
            val expected = KoblasEngine.scalar.sum(entries, 0, entries.size)

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
                    KoblasEngine.scalar.sum(entries, 0, entries.size)
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
        val expected = KoblasEngine.scalar.sum(values, 0, values.size)

        assertSumAgreesWithReference(expected, a.sum())
        assertVectorAgreesWithReference(values, a.sum(0), DoubleArray(2))
    }

    private fun reductionMatrices(scale: Double = 1.0): List<Matrix> {
        val sparse = SparseMatrix.ofColumns(
            3,
            4,
            listOf(
                listOf(0 to 2.5, 1 to -3.0, 2 to -1.0),
                listOf(0 to 0.0, 1 to 4.0),
                emptyList(),
                listOf(1 to -2.0, 2 to 0.75),
            ),
        )
        for (k in sparse.values.indices) sparse.values[k] *= scale
        val dense = sparse.denseCopy()
        val view = paddedRegion(3, 4)
        for (j in 0 until view.cols) for (i in 0 until view.rows) view[i, j] = dense[i, j]
        return listOf(dense, view, sparse, ForeignMatrix(dense))
    }

    private class ForeignMatrix(private val source: Matrix) : Matrix {
        override val rows: Int get() = source.rows
        override val cols: Int get() = source.cols
        override fun get(i: Int, j: Int): Double = source[i, j]
        override fun set(i: Int, j: Int, v: Double) {
            source[i, j] = v
        }
    }

    private fun assertSumAgreesWithReference(expected: Double, actual: Double) {
        assertClose(expected, actual, "matrix sum")
    }

    private fun assertVectorAgreesWithReference(expected: DoubleArray, actual: DenseVector, before: DoubleArray) {
        val backing = before.copyOf()
        for (i in expected.indices) backing[actual.offset + i * actual.stride] = expected[i]
        assertClose(backing, actual.values, "vector backing buffer")
    }
}
