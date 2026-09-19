package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.prepare
import com.eignex.koblas.times
import com.eignex.koblas.transpose
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Shapes at the edge of what a CSC result can represent.
 *
 * A validated sparse matrix bounds its own column count, because holding one pointer per column plus a final
 * one is what makes it a matrix. Its row count is bounded by nothing, so an operation whose output has one
 * column per input row is where a row count that cannot be represented first becomes an array length. Every
 * fixture here is a tall empty matrix: it declares an enormous row count and stores nothing, so the shapes
 * are reached without allocating anything larger than a handful of entries.
 */
class SparseCapacityTest {

    private fun tallEmpty(rows: Int, cols: Int = 0): SparseMatrix =
        SparseMatrix.wrap(rows, cols, IntArray(cols + 1), IntArray(0), DoubleArray(0))

    @Test
    fun `a transpose whose columns cannot be counted is rejected rather than wrapped`() {
        val source = tallEmpty(Int.MAX_VALUE)

        val failure = assertFailsWith<DimensionMismatch> { source.transpose() }

        assertTrue(failure.message!!.contains("more than one array can hold"), failure.message!!)
    }

    @Test
    fun `an ordinary transpose is unaffected by the representability check`() {
        val source = tallEmpty(rows = 3, cols = 2)

        val transposed = source.transpose()

        assertEquals(2, transposed.rows)
        assertEquals(3, transposed.cols)
        assertEquals(0, transposed.nnz)
        assertContentEquals(intArrayOf(0, 0, 0, 0), transposed.copyColumnPointers())
    }

    /**
     * A product that can reach no position discovers no structure, and the scratch that would have found it
     * is indexed by the result's rows. Allocating that for a provably empty result is what would turn a valid
     * tall shape into an out-of-memory error rather than an empty matrix.
     */
    @Test
    fun `an empty product of a tall operand returns the shape without row-sized scratch`() {
        val tall = tallEmpty(Int.MAX_VALUE)
        val square = tallEmpty(rows = 0, cols = 0)

        val product = tall * square

        assertEquals(Int.MAX_VALUE, product.rows)
        assertEquals(0, product.cols)
        assertEquals(0, product.nnz)
        assertContentEquals(intArrayOf(0), product.copyColumnPointers())
    }

    @Test
    fun `a transposed rank product of a tall operand has the order its columns give it`() {
        val tall = tallEmpty(Int.MAX_VALUE)

        val transposed = koblas.syrk(tall, transpose = true)

        assertEquals(0, transposed.rows)
        assertEquals(0, transposed.cols)
        assertEquals(0, transposed.nnz)
    }

    @Test
    fun `a rank product whose order cannot be counted is rejected rather than wrapped`() {
        val tall = tallEmpty(Int.MAX_VALUE)

        val failure = assertFailsWith<DimensionMismatch> { koblas.syrk(tall, transpose = false) }

        assertTrue(failure.message!!.contains("more than one array can hold"), failure.message!!)
    }

    /**
     * The dense-destination rank update never builds a result of its own, so its limit is the scratch indexed
     * by the source's rows. A source with nothing stored reaches no position, and the destination it was
     * given is the whole of the answer.
     */
    @Test
    fun `a dense rank update over a tall empty operand scales the triangle and stops`() {
        val tall = tallEmpty(Int.MAX_VALUE)
        val destination = DenseMatrix.wrap(0, 0, DoubleArray(0))

        koblas.syrk(1.0, tall, transpose = true, 0.5, destination, workspace = Workspace())

        assertEquals(0, destination.values.size)
    }

    @Test
    fun `a dense rank update over an empty source still applies beta to the selected triangle`() {
        val empty = SparseMatrix.ofColumns(2, 0, emptyList())
        val destination = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        koblas.syrk(1.0, empty, transpose = false, 2.0, destination, lower = true)

        // The upper triangle is untouched, and the lower one is doubled.
        assertContentEquals(doubleArrayOf(2.0, 4.0, 3.0, 8.0), destination.values)
    }

    /**
     * The shortcut that avoids row-sized scratch must not swallow a product that has structure to find. Both
     * operands store an entry here, so the traversal runs and the position their patterns meet at survives.
     */
    @Test
    fun `the empty-result shortcut does not swallow a product that has structure`() {
        val a = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 1.0)))
        val b = SparseMatrix.ofColumns(1, 2, listOf(emptyList(), listOf(0 to 2.0)))

        val product = a * b

        assertEquals(2, product.rows)
        assertEquals(2, product.cols)
        assertEquals(1, product.nnz)
        assertEquals(2.0, product[0, 1])
    }

    /**
     * A prepared transposed product over the same tall shape. The result is empty and perfectly valid, so the
     * snapshot must reach it without deriving a transpose whose pointer array could not exist.
     */
    @Test
    fun `a prepared transposed product of a tall operand reaches its empty result`() {
        val prepared = tallEmpty(Int.MAX_VALUE).prepare()
        val destination = DenseMatrix.zero(0, 0)

        prepared.gemm(1.0, transposeA = true, b = DenseMatrix.zero(Int.MAX_VALUE, 0), beta = 0.0, c = destination)

        assertEquals(0, destination.values.size)
    }

    @Test
    fun `a prepared transposed sparse product of a tall operand reaches its empty result`() {
        val prepared = tallEmpty(Int.MAX_VALUE).prepare()

        val product = prepared.gemm(1.0, transposeA = true, b = tallEmpty(Int.MAX_VALUE), transposeB = false)

        assertEquals(0, product.rows)
        assertEquals(0, product.cols)
        assertEquals(0, product.nnz)
    }

    @Test
    fun `an unprepared transposed product of a tall operand reaches its empty result`() {
        val tall = tallEmpty(Int.MAX_VALUE)

        val product = koblas.gemm(1.0, tall, true, tall, false)

        assertEquals(0, product.rows)
        assertEquals(0, product.cols)
        assertEquals(0, product.nnz)
    }

    @Test
    fun `a scaled addition of a tall operand keeps its shape without orienting it`() {
        val tall = tallEmpty(Int.MAX_VALUE)

        val sum = koblas.addScaled(1.0, tall, transposeA = false, b = tall)

        assertEquals(Int.MAX_VALUE, sum.rows)
        assertEquals(0, sum.cols)
        assertEquals(0, sum.nnz)
    }
}
