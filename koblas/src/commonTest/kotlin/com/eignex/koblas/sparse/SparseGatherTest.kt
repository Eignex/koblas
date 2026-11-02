package com.eignex.koblas.sparse

import com.eignex.koblas.DenseVector
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.SparseVector
import com.eignex.koblas.gather
import com.eignex.koblas.gatherZero
import kotlin.test.Test
import kotlin.test.assertFailsWith

class SparseGatherTest {

    private fun pattern() = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(9.0, 9.0))

    @Test
    fun `gather rejects a dense vector of another length`() {
        assertFailsWith<DimensionMismatch> { gather(pattern(), DenseVector.of(DoubleArray(7))) }
    }

    @Test
    fun `gatherZero rejects a dense vector of another length`() {
        assertFailsWith<DimensionMismatch> { gatherZero(pattern(), DenseVector.of(DoubleArray(7))) }
    }
}
