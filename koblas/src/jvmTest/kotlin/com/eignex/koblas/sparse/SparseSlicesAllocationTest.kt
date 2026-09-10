package com.eignex.koblas.sparse

import com.eignex.koblas.koblas
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

class SparseSlicesAllocationTest {
    @Test
    fun `warmed sparse slice operations allocate nothing`() {
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
        val arithmeticStatus = IntArray(1)
        val dense = DoubleArray(dimension) { it * 0.01 + 1.0 }
        val rawDestination = DoubleArray(dimension)
        val clearValues = DoubleArray(dimension) { 1.0 }
        val clearMarks = IntArray(dimension) { 1 }
        var touchedCount = SparseSlices.scatterAxpy(
            1.0, indices, 0, values, 0, count,
            accumulator, marks, 11, touched, 0, 0,
        )
        repeat(200) {
            SparseSlices.gatherClearTouched(
                touched, 0, touchedCount, accumulator, marks,
                outIndices, 0, outValues, 0, compactExactZeros = true,
            )
            touchedCount = 0
            touchedCount = SparseSlices.scatterAxpy(
                1e-12, indices, 0, values, 0, count,
                accumulator, marks, 11, touched, 0, touchedCount,
            )
            SparseSlices.scatterAxpyChecked(
                1e-12, indices, 0, values, 0, count,
                accumulator, marks, 11, touched, 0, touchedCount, arithmeticStatus, 0,
            )
            SparseSlices.gatherTouched(
                touched, 0, touchedCount, accumulator, outIndices, 0, outValues, 0, compactExactZeros = true,
            )
            val maximum = SparseSlices.activeColumnMaxAbs(indices, 0, values, 0, count, active)
            SparseSlices.pivotCandidatePositions(
                indices, 0, values, 0, count, active, maximum, 0.0, 0.1, outIndices, 0,
            )
            koblas.sparseKernels.dot(indices, 0, values, 0, count, dense)
            koblas.sparseKernels.axpy(rawDestination, 1e-12, indices, 0, values, 0, count)
            koblas.sparseKernels.scatter(indices, 0, values, 0, count, rawDestination)
            koblas.sparseKernels.nrm2(indices, 0, count, dense)
            SparseSlices.reduceDotChecked(
                0.0, false, indices, 0, values, 0, count, dense, arithmeticStatus, 0,
            )
            SparseSlices.clearTouched(indices, 0, count, clearValues, clearMarks)
        }

        val bytes = bytesPerIteration(2_000) {
            SparseSlices.gatherClearTouched(
                touched, 0, touchedCount, accumulator, marks,
                outIndices, 0, outValues, 0, compactExactZeros = true,
            )
            touchedCount = 0
            touchedCount = SparseSlices.scatterAxpy(
                1e-12, indices, 0, values, 0, count,
                accumulator, marks, 11, touched, 0, touchedCount,
            )
            SparseSlices.scatterAxpyChecked(
                1e-12, indices, 0, values, 0, count,
                accumulator, marks, 11, touched, 0, touchedCount, arithmeticStatus, 0,
            )
            SparseSlices.gatherTouched(
                touched, 0, touchedCount, accumulator, outIndices, 0, outValues, 0, compactExactZeros = true,
            )
            val maximum = SparseSlices.activeColumnMaxAbs(indices, 0, values, 0, count, active)
            SparseSlices.pivotCandidatePositions(
                indices, 0, values, 0, count, active, maximum, 0.0, 0.1, outIndices, 0,
            )
            koblas.sparseKernels.dot(indices, 0, values, 0, count, dense)
            koblas.sparseKernels.axpy(rawDestination, 1e-12, indices, 0, values, 0, count)
            koblas.sparseKernels.scatter(indices, 0, values, 0, count, rawDestination)
            koblas.sparseKernels.nrm2(indices, 0, count, dense)
            SparseSlices.reduceDotChecked(
                0.0, false, indices, 0, values, 0, count, dense, arithmeticStatus, 0,
            )
            SparseSlices.clearTouched(indices, 0, count, clearValues, clearMarks)
        }

        assertTrue(bytes <= 64.0, "sparse slice operations allocated $bytes B per iteration")
    }
}
