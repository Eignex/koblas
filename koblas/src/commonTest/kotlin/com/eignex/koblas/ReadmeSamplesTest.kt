package com.eignex.koblas

import com.eignex.koblas.dense.*
import kotlin.test.Test
import kotlin.test.assertEquals

class ReadmeSamplesTest {

    @Test
    fun `the triplet sample builds the same matrix the column sample does`() {
        val s = SparseMatrix.ofTriplets(
            rows = 2,
            cols = 2,
            rowIdx = intArrayOf(0, 1, 0, 1),
            colIdx = intArrayOf(0, 0, 1, 1),
            values = doubleArrayOf(2.0, 1.0, 1.0, 3.0),
        )
        val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
        assertEquals(
            s,
            SparseMatrix.ofColumns(2, 2, cols),
            "README triplet sample should match the column one",
        )
    }

    @Test
    fun `koblasInfo has the shape the sample shows`() {
        assertEquals("engine=${koblas.name}", koblasInfo)
    }
}
