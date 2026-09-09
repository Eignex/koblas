package com.eignex.koblas.sparse

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

@OptIn(ExperimentalKoblasApi::class)
class SparseWorkspaceAllocationTest {
    @Test
    fun `warmed sparse workspace operations allocate nothing`() {
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
        var touchedCount = SparseWorkspace.scatterAxpy(
            1.0, indices, 0, values, 0, count,
            accumulator, marks, 11, touched, 0, 0,
        )
        repeat(200) {
            SparseWorkspace.gatherClearTouched(
                touched, 0, touchedCount, accumulator, marks,
                outIndices, 0, outValues, 0, compactExactZeros = true,
            )
            touchedCount = 0
            touchedCount = SparseWorkspace.scatterAxpy(
                1e-12, indices, 0, values, 0, count,
                accumulator, marks, 11, touched, 0, touchedCount,
            )
            SparseWorkspace.gatherTouched(
                touched, 0, touchedCount, accumulator, outIndices, 0, outValues, 0, compactExactZeros = true,
            )
            val maximum = SparseWorkspace.activeColumnMaxAbs(indices, 0, values, 0, count, active)
            SparseWorkspace.pivotCandidatePositions(
                indices, 0, values, 0, count, active, maximum, 0.0, 0.1, outIndices, 0,
            )
        }

        val bytes = bytesPerIteration(2_000) {
            SparseWorkspace.gatherClearTouched(
                touched, 0, touchedCount, accumulator, marks,
                outIndices, 0, outValues, 0, compactExactZeros = true,
            )
            touchedCount = 0
            touchedCount = SparseWorkspace.scatterAxpy(
                1e-12, indices, 0, values, 0, count,
                accumulator, marks, 11, touched, 0, touchedCount,
            )
            SparseWorkspace.gatherTouched(
                touched, 0, touchedCount, accumulator, outIndices, 0, outValues, 0, compactExactZeros = true,
            )
            val maximum = SparseWorkspace.activeColumnMaxAbs(indices, 0, values, 0, count, active)
            SparseWorkspace.pivotCandidatePositions(
                indices, 0, values, 0, count, active, maximum, 0.0, 0.1, outIndices, 0,
            )
        }

        assertTrue(bytes <= 64.0, "sparse workspace operations allocated $bytes B per iteration")
    }
}
