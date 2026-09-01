package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import kotlin.test.*

/** The sparse half of the operator surface, against the dense operators it mirrors. */
class SparseOperatorsTest {

    private fun sparseMatrix(vararg columns: List<Pair<Int, Double>>): F64SparseMatrix =
        F64SparseMatrix.ofColumns(3, columns.size, columns.toList())

    private fun densified(a: F64SparseMatrix): F64DenseMatrix = F64DenseMatrix(a.rows, a.cols).also { out ->
        for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> out[i, j] = v }
    }

    private fun sparseVector(size: Int, vararg pairs: Pair<Int, Double>): F64SparseVector =
        F64SparseVector.of(size, pairs.map { it.first }.toIntArray(), pairs.map { it.second }.toDoubleArray())

    @Test
    fun `scaling a sparse matrix keeps its pattern and matches the dense product`() {
        val a = sparseMatrix(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 0.0), listOf())

        val scaled = a * 1.5

        assertContentEquals(a.copyColumnPointers(), scaled.copyColumnPointers(), "column pointers")
        assertContentEquals(a.copyRowIndices(), scaled.copyRowIndices(), "row indices")
        assertClose(densified(a) * 1.5, densified(scaled), "alpha times A")
        assertClose(densified(a) * 1.5, densified(1.5 * a), "A times alpha")
        assertClose(-densified(a), densified(-a), "unary minus")
    }

    @Test
    fun `a zero scale keeps every stored position`() {
        val a = sparseMatrix(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0), listOf())

        val scaled = a * 0.0

        assertEquals(a.nnz, scaled.nnz, "a scaled pattern must not be pruned")
        // Scaling a negative entry by zero gives -0.0, which is still the zero this stores.
        for (value in scaled.values) assertEquals(0.0, value, 0.0, "every scaled entry")
    }

    @Test
    fun `sparse addition and subtraction match the dense operators`() {
        val a = sparseMatrix(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0), listOf())
        val b = sparseMatrix(listOf(1 to 0.5), listOf(1 to -3.0), listOf(2 to 4.0))

        assertClose(densified(a) + densified(b), densified(a + b), "A plus B")
        assertClose(densified(a) - densified(b), densified(a - b), "A minus B")
    }

    @Test
    fun `a sparse sum stores the union of the patterns including a cancellation`() {
        val a = sparseMatrix(listOf(0 to 2.0), listOf(1 to 3.0), listOf())
        val b = sparseMatrix(listOf(1 to 0.5), listOf(1 to -3.0), listOf())

        val sum = a + b

        assertContentEquals(intArrayOf(0, 2, 3, 3), sum.copyColumnPointers(), "union column pointers")
        assertContentEquals(intArrayOf(0, 1, 1), sum.copyRowIndices(), "union row indices")
        assertEquals(0.0, sum[1, 1], "the cancelled entry stays stored at its zero value")
    }

    @Test
    fun `mismatched sparse shapes are rejected`() {
        val a = sparseMatrix(listOf(0 to 1.0))
        val wider = sparseMatrix(listOf(0 to 1.0), listOf())

        assertFailsWith<DimensionMismatch> { a + wider }
        assertFailsWith<DimensionMismatch> { a - wider }
        assertFailsWith<DimensionMismatch> { sparseVector(3, 0 to 1.0) + sparseVector(4, 0 to 1.0) }
    }

    @Test
    fun `sparse vector operators match their dense equivalents`() {
        val x = sparseVector(5, 0 to 2.0, 3 to -1.0)
        val y = sparseVector(5, 3 to 4.0, 4 to 0.5)
        val xDense = F64DenseVector.of(x.toDoubleArray())
        val yDense = F64DenseVector.of(y.toDoubleArray())

        assertClose((xDense * 1.5).data, (x * 1.5).toDoubleArray(), "alpha times x")
        assertClose((1.5 * xDense).data, (1.5 * x).toDoubleArray(), "x times alpha")
        assertClose((-xDense).data, (-x).toDoubleArray(), "unary minus")
        assertClose((xDense + yDense).data, (x + y).toDoubleArray(), "x plus y")
        assertClose((xDense - yDense).data, (x - y).toDoubleArray(), "x minus y")
    }

    @Test
    fun `a sparse vector sum stores the union of the stored positions`() {
        val sum = sparseVector(5, 0 to 2.0, 3 to -1.0) + sparseVector(5, 3 to 1.0, 4 to 0.5)

        assertContentEquals(intArrayOf(0, 3, 4), sum.indices, "union positions")
        assertEquals(0.0, sum[3], "the cancelled entry stays stored at its zero value")
    }

    @Test
    fun `withRankOne matches the dense ger over the same operands`() {
        val a = sparseMatrix(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0), listOf())
        val x = sparseVector(3, 0 to 1.5, 2 to -2.0)
        val y = F64DenseVector.of(doubleArrayOf(0.5, 0.0, 2.0))
        val expected = densified(a)
        expected.ger(-0.75, x, y)

        assertClose(expected, densified(a.withRankOne(-0.75, x, y)), "sparse ger")
    }

    @Test
    fun `withRankOne leaves the source untouched and rejects a mismatched shape`() {
        val a = sparseMatrix(listOf(0 to 2.0), listOf(), listOf())
        val before = a.values.copyOf()

        a.withRankOne(1.0, F64DenseVector.of(doubleArrayOf(1.0, 1.0, 1.0)), F64DenseVector.of(DoubleArray(3) { 1.0 }))

        assertContentEquals(before, a.values, "the source must not be mutated")
        assertFailsWith<DimensionMismatch> {
            a.withRankOne(1.0, F64DenseVector.zero(2), F64DenseVector.of(DoubleArray(3) { 1.0 }))
        }
    }

    @Test
    fun `a zero alpha copies the source rather than updating it`() {
        val a = sparseMatrix(listOf(0 to 2.0, 2 to -1.0), listOf(1 to 3.0), listOf())
        val x = F64DenseVector.of(doubleArrayOf(1.0, 1.0, 1.0))

        val copy = a.withRankOne(0.0, x, F64DenseVector.of(DoubleArray(3) { 1.0 }))

        assertContentEquals(a.copyRowIndices(), copy.copyRowIndices(), "row indices")
        assertContentEquals(a.values, copy.values, "values")
        assertTrue(a.values !== copy.values, "the copy must own its values")
    }
}
