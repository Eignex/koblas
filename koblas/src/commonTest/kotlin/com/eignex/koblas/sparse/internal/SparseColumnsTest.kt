package com.eignex.koblas.sparse.internal

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.withColumn
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

class SparseColumnsTest {

    private val a = F64SparseMatrix.ofColumns(
        rows = 4,
        cols = 3,
        columns = listOf(
            listOf(0 to 1.0, 3 to 2.0),
            listOf(1 to -1.0),
            listOf(0 to 0.5, 2 to 4.0),
        ),
    )

    private fun column(vararg pairs: Pair<Int, Double>) =
        F64SparseVector.of(4, pairs.map { it.first }.toIntArray(), pairs.map { it.second }.toDoubleArray())

    @Test
    fun `replacing columns in one pass matches replacing them one at a time`() {
        val first = column(1 to 7.0)
        val second = column(0 to -3.0, 2 to 8.0)

        val batched = replaceColumns(a, mapOf(2 to second, 0 to first))

        assertEquals(a.withColumn(0, first).withColumn(2, second), batched)
    }

    /** A driver may pivot the same slot more than once, and only the last column it put there is the basis. */
    @Test
    fun `the last replacement of a column is the one that lands`() {
        val early = column(1 to 7.0)
        val late = column(2 to -5.0, 3 to 6.0)
        val pending = LinkedHashMap<Int, F64SparseVector>()
        pending[1] = early
        pending[1] = late

        val batched = replaceColumns(a, pending)

        assertEquals(a.withColumn(1, early).withColumn(1, late), batched)
    }

    @Test
    fun `an entering column that stores a zero keeps it`() {
        val stored = column(0 to 0.0, 2 to 1.0)

        val batched = replaceColumns(a, mapOf(1 to stored))

        assertEquals(a.withColumn(1, stored), batched)
        assertEquals(a.nnz - 1 + 2, batched.nnz, "the stored zero was dropped")
    }

    @Test
    fun `no replacements is the matrix itself`() {
        assertSame(a, replaceColumns(a, emptyMap()))
    }

    @Test
    fun `a snapshot does not follow later writes to the vector it was taken from`() {
        val live = column(1 to 7.0)
        val kept = live.snapshot()

        live.values[0] = 9.0

        assertEquals(7.0, kept.values[0], 0.0)
        assertEquals(replaceColumns(a, mapOf(0 to kept)), a.withColumn(0, column(1 to 7.0)))
    }
}
