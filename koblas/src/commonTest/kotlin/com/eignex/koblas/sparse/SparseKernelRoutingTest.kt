package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.dense.ScalarKernels
import com.eignex.koblas.dense.ScalarPanelKernels
import kotlin.test.Test
import kotlin.test.assertEquals

class SparseKernelRoutingTest {
    private class RecordingIndexed : IndexedSparseKernels by ScalarIndexedSparseKernels {
        var dots = 0
        var updates = 0

        override fun dotDense(
            indices: IntArray,
            values: DoubleArray,
            fromIndex: Int,
            toIndex: Int,
            dense: DoubleArray,
        ): Double {
            dots++
            return ScalarIndexedSparseKernels.dotDense(indices, values, fromIndex, toIndex, dense)
        }

        override fun axpy(
            indices: IntArray,
            values: DoubleArray,
            fromIndex: Int,
            toIndex: Int,
            alpha: Double,
            destination: DoubleArray,
        ) {
            updates++
            ScalarIndexedSparseKernels.axpy(indices, values, fromIndex, toIndex, alpha, destination)
        }
    }

    private val matrix = SparseMatrix.ofColumns(
        3,
        3,
        listOf(
            listOf(0 to 2.0, 1 to -1.0),
            listOf(1 to 3.0, 2 to 0.5),
            listOf(2 to 4.0),
        ),
    )

    @Test
    fun `gemv routes each csc column through selected indexed leaves`() {
        val indexed = RecordingIndexed()
        val panels = PortableSparsePanelKernels(ScalarKernels, ScalarPanelKernels)
        val algorithms = SparseAlgorithms(ScalarKernels, indexed, panels)

        algorithms.gemv(1.0, matrix, doubleArrayOf(1.0, 2.0, 3.0), 0.0, DoubleArray(3), transpose = false)
        algorithms.gemv(1.0, matrix, doubleArrayOf(1.0, 2.0, 3.0), 0.0, DoubleArray(3), transpose = true)

        assertEquals(matrix.cols, indexed.updates)
        assertEquals(matrix.cols, indexed.dots)
    }
}
