package com.eignex.koblas.internal.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.times
import com.eignex.koblas.transpose
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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

        koblas.syrk(1.0, tall, transpose = true, 0.5, destination, scratch = Scratch())

        assertEquals(0, destination.values.size)
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
