package com.eignex.koblas.internal.storage.sparse

import com.eignex.koblas.*
import kotlin.test.Test
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

    // The dense-destination rank update builds no result of its own, so its limit is the scratch indexed by
    // the source's rows.
    @Test
    fun `a dense rank update over a tall empty operand scales the triangle and stops`() {
        val tall = tallEmpty(Int.MAX_VALUE)
        val destination = DenseMatrix.wrap(0, 0, DoubleArray(0))

        koblas.syrk(1.0, tall, transpose = true, 0.5, destination, scratch = Scratch())

        assertEquals(0, destination.values.size)
    }
}
