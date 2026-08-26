package com.eignex.koblas.sparse

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.gather
import com.eignex.koblas.gatherZero
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparseGatherTest {

    private fun pattern() = F64SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(9.0, 9.0))

    private fun dense() = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @Test
    fun `gather takes the dense entries at the stored positions`() {
        val x = pattern()

        F64ReferenceSparseLinearAlgebra.gather(x, dense())

        assertTrue(doubleArrayOf(2.0, 5.0).contentEquals(x.values), "gathered ${x.values.toList()}")
    }

    @Test
    fun `gather leaves the dense vector it read untouched`() {
        val from = dense()

        F64ReferenceSparseLinearAlgebra.gather(pattern(), from)

        assertTrue(dense().contentEquals(from), "gather wrote back into its source")
    }

    @Test
    fun `gather keeps the pattern where the dense vector stores more`() {
        val x = pattern()

        F64ReferenceSparseLinearAlgebra.gather(x, dense())

        assertTrue(intArrayOf(1, 4).contentEquals(x.copyIndices()), "the pattern must survive a gather")
    }

    @Test
    fun `gatherZero empties the dense vector at exactly the positions it read`() {
        val from = dense()

        F64ReferenceSparseLinearAlgebra.gatherZero(pattern(), from)

        assertTrue(
            doubleArrayOf(1.0, 0.0, 3.0, 4.0, 0.0, 6.0).contentEquals(from),
            "left ${from.toList()}",
        )
    }

    @Test
    fun `gatherZero takes the same values a gather would`() {
        val gathered = pattern().also { F64ReferenceSparseLinearAlgebra.gather(it, dense()) }
        val zeroed = pattern().also { F64ReferenceSparseLinearAlgebra.gatherZero(it, dense()) }

        assertTrue(gathered.values.contentEquals(zeroed.values), "the two gathers must agree on the values")
    }

    @Test
    fun `gather undoes a scatter over the stored pattern`() {
        val x = F64SparseVector.of(6, intArrayOf(0, 3, 5), doubleArrayOf(-1.5, 2.5, 7.0))
        val expected = x.values.copyOf()
        val dense = DoubleArray(6)
        F64ReferenceSparseLinearAlgebra.scatter(x, dense)

        F64ReferenceSparseLinearAlgebra.gather(x, dense)

        assertTrue(expected.contentEquals(x.values), "round trip gave ${x.values.toList()}")
    }

    @Test
    fun `gather rejects a dense vector of another length`() {
        assertFailsWith<DimensionMismatch> { gather(pattern(), F64DenseVector.of(DoubleArray(7))) }
    }

    @Test
    fun `gatherZero rejects a dense vector of another length`() {
        assertFailsWith<DimensionMismatch> { gatherZero(pattern(), F64DenseVector.of(DoubleArray(7))) }
    }
}
