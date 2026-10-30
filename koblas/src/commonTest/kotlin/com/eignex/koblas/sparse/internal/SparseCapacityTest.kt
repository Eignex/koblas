package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.gemm
import com.eignex.koblas.koblas
import com.eignex.koblas.times
import com.eignex.koblas.transpose
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame
import kotlin.test.assertTrue

/**
 * CSC bounds its column count but not its row count. Tall empty fixtures expose operations
 * that turn an unrepresentable row count into a pointer-array length.
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

    // The scratch that would find structure is indexed by the result's rows, so allocating it for a provably
    // empty result turns a valid tall shape into an out-of-memory error.
    @Test
    fun `an empty product of a tall operand returns the shape without row sized scratch`() {
        val tall = tallEmpty(Int.MAX_VALUE)
        val square = tallEmpty(rows = 0, cols = 0)

        val product = tall * square

        assertEquals(Int.MAX_VALUE, product.rows)
        assertEquals(0, product.cols)
        assertEquals(0, product.nnz)
        assertContentEquals(intArrayOf(0), product.copyColumnPointers())
    }

    @Test
    fun `nonempty disjoint operands return a tall empty product without row sized scratch`() {
        val a = SparseMatrix.wrap(Int.MAX_VALUE, 2, intArrayOf(0, 1, 1), intArrayOf(0), doubleArrayOf(1.0))
        val b = SparseMatrix.wrap(2, 1, intArrayOf(0, 1), intArrayOf(1), doubleArrayOf(2.0))

        val product = a * b

        assertEquals(Int.MAX_VALUE, product.rows)
        assertEquals(1, product.cols)
        assertEquals(0, product.nnz)
        assertContentEquals(intArrayOf(0, 0), product.copyColumnPointers())
        assertContentEquals(intArrayOf(), product.copyRowIndices())
        assertContentEquals(doubleArrayOf(), product.values)
    }

    @Test
    fun `disjoint products own fresh empty arrays`() {
        val a = SparseMatrix.wrap(3, 2, intArrayOf(0, 1, 1), intArrayOf(0), doubleArrayOf(1.0))
        val b = SparseMatrix.wrap(2, 2, intArrayOf(0, 0, 1), intArrayOf(1), doubleArrayOf(2.0))

        val first = a * b
        val second = a * b

        assertNotSame(a.colPointers, first.colPointers)
        assertNotSame(b.colPointers, first.colPointers)
        assertNotSame(first.colPointers, second.colPointers)
        assertNotSame(first.rowIndices, second.rowIndices)
        assertNotSame(first.values, second.values)
        assertContentEquals(intArrayOf(0, 0, 0), first.copyColumnPointers())
    }

    @Test
    fun `a later reached column keeps stored zeros in the product pattern`() {
        for ((leftValue, rightValue) in listOf(0.0 to 2.0, 1.0 to 0.0)) {
            val a = SparseMatrix.wrap(3, 2, intArrayOf(0, 0, 1), intArrayOf(2), doubleArrayOf(leftValue))
            val b = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(0, 1), doubleArrayOf(1.0, rightValue))

            val product = a * b

            assertEquals(3, product.rows)
            assertEquals(2, product.cols)
            assertContentEquals(intArrayOf(0, 0, 1), product.copyColumnPointers())
            assertContentEquals(intArrayOf(2), product.copyRowIndices())
            assertContentEquals(doubleArrayOf(0.0), product.values)
        }
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

    // The dense-destination rank update builds no result of its own, so its limit is the scratch indexed by
    // the source's rows.
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

        assertContentEquals(doubleArrayOf(2.0, 4.0, 3.0, 8.0), destination.values)
    }

    // The shortcut that avoids row-sized scratch must not swallow a product with structure to find.
    @Test
    fun `the empty result shortcut does not swallow a product that has structure`() {
        val a = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 1.0)))
        val b = SparseMatrix.ofColumns(1, 2, listOf(emptyList(), listOf(0 to 2.0)))

        val product = a * b

        assertEquals(2, product.rows)
        assertEquals(2, product.cols)
        assertEquals(1, product.nnz)
        assertEquals(2.0, product[0, 1])
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
