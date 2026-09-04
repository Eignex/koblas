package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * What reading a Markowitz factor's triangles costs in allocation.
 *
 * The factors are `get()` properties that rebuild on every read, so a caller holding a factorization and
 * reading `l` twice pays twice. What this pins is the part that is not the arrays themselves: the ordering
 * each column needs must not go through a boxed list.
 */
class SparseFactorReadAllocationTest {

    private fun factorization(order: Int): F64SparseLuFactorization {
        val a = sparseConformanceSystem(order, Random(20260904))
        return F64ReferenceSparseLinearAlgebra.factor(a)
    }

    /**
     * The bound is the CSC arrays a read hands back, with room to spare: two index arrays, one value array
     * and their headers. A boxed ordering per column costs an `ArrayList` and an `Integer` per entry on top
     * of that, which is what puts a read over this.
     */
    @Test
    fun `reading a factor triangle allocates little beyond the arrays it returns`() {
        val order = 64
        val lu = factorization(order)
        val entries = lu.l.nnz + lu.u.nnz

        val bytes = bytesPerIteration(iterations = 200) { readBoth(lu) }

        val arrays = (entries * (Int.SIZE_BYTES + Double.SIZE_BYTES) + order * 2 * Int.SIZE_BYTES).toDouble()
        assertTrue(
            bytes < arrays * 1.2,
            "reading l and u allocated $bytes bytes for $entries entries, against $arrays for the arrays",
        )
    }

    private fun readBoth(lu: F64SparseLuFactorization): Int {
        val l: F64SparseMatrix = lu.l
        val u: F64SparseMatrix = lu.u
        return l.nnz + u.nnz
    }
}
