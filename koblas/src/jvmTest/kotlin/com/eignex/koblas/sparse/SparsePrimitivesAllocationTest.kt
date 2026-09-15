package com.eignex.koblas.sparse

import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

class SparsePrimitivesAllocationTest {
    @Test
    fun `warmed sparse primitives allocate nothing`() {
        val dimension = 128
        val count = 24
        val indices = IntArray(count) { (it * 5 + 3) % dimension }
        val values = DoubleArray(count) { it * 0.125 - 1.0 }
        val accumulator = DoubleArray(dimension)
        val marks = IntArray(dimension)
        val touched = IntArray(dimension)
        val outIndices = IntArray(dimension)
        val outValues = DoubleArray(dimension)
        val active = BooleanArray(dimension) { it % 3 != 0 }
        val status = IntArray(1)

        fun cycle() {
            var touchedCount = SparsePrimitives.scatterWorkspace(
                1e-12, indices, 0, values, 0, count, accumulator, marks, 11, touched, 0, 0,
            )
            touchedCount = SparsePrimitives.scatterWorkspaceChecked(
                1e-12, indices, 0, values, 0, count,
                accumulator, marks, 11, touched, 0, touchedCount, status, 0, 1, 2,
            )
            SparsePrimitives.gatherWorkspace(
                touched, 0, touchedCount, accumulator, outIndices, 0, outValues, 0,
                compactExactZeros = true, marks = null,
            )
            SparsePrimitives.gatherWorkspace(
                touched, 0, touchedCount, accumulator, outIndices, 0, outValues, 0,
                compactExactZeros = true, marks = marks,
            )
            SparsePrimitives.clearWorkspace(touched, 0, touchedCount, accumulator, marks)
            val maximum = SparsePrimitives.activeMaximum(indices, 0, values, 0, count, active)
            SparsePrimitives.selectPivotCandidates(
                indices, 0, values, 0, count, active, 0.0, 0.1 * maximum, outIndices, 0,
            )
        }

        repeat(200) { cycle() }
        val bytes = bytesPerIteration(2_000) { cycle() }

        assertTrue(bytes <= 64.0, "sparse primitives allocated $bytes B per iteration")
    }
}
