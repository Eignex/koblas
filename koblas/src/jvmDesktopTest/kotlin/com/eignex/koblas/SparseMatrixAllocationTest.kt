package com.eignex.koblas

import com.eignex.koblas.testutil.allocation.ARRAY_HEADER_BYTES
import com.eignex.koblas.testutil.allocation.AllocationCase
import com.eignex.koblas.testutil.allocation.FLOOR_BYTES
import com.eignex.koblas.testutil.allocation.parameterRows
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.Test

/** Triplet construction reuses its cursor and full result arrays, compacting only when entries repeat. */
@RunWith(Parameterized::class)
class SparseMatrixAllocationTest internal constructor(private val case: AllocationCase) {
    @Test
    fun `warmed calls stay within their allocation budget`() = case.assertWithinBudget()

    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> = parameterRows(intArrayOf(1, 2).map(::tripletCase))

        private fun tripletCase(copies: Int): AllocationCase {
            val rows = 128
            val columns = 4
            val unique = 32
            val count = copies * unique
            val indices = IntArray(count) { (it / copies % 8) * 4 }
            val columnIndices = IntArray(count) { it / copies / 8 }
            val values = DoubleArray(count) { 1.0 + it * 0.125 }
            val rowScratch = ARRAY_HEADER_BYTES + (rows + 1) * Int.SIZE_BYTES + 4
            val pointers = ARRAY_HEADER_BYTES + (columns + 1) * Int.SIZE_BYTES + 4
            val runs = 2 * (2 * ARRAY_HEADER_BYTES + count * (Int.SIZE_BYTES + Double.SIZE_BYTES))
            val compact = if (copies == 1) 0 else 2 * ARRAY_HEADER_BYTES + unique * (Int.SIZE_BYTES + Double.SIZE_BYTES)
            // Two run descriptors, the result object, alignment and the probe's fixed allowance.
            val budget = rowScratch + pointers + runs + compact + 2 * FLOOR_BYTES

            return AllocationCase("triplet construction with $copies copies of each entry", 1_000, budget) {
                SparseMatrix.ofTriplets(rows, columns, indices, columnIndices, values)
            }
        }
    }
}
