package com.eignex.koblas.sparse

import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.dense.ScalarVectorKernels
import com.eignex.koblas.internal.operands.denseOperand
import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.internal.sparse.SparseKernelAdapter
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

class SparseKernelAdapterTest {
    @Test
    fun `column dot lends the original arrays at the selected offsets to the indexed leaf`() {
        val matrix = SparseMatrix.wrap(
            5,
            4,
            intArrayOf(0, 1, 3, 3, 3),
            intArrayOf(4, 0, 3),
            doubleArrayOf(9.0, 2.0, -3.0),
        )
        val denseOperand = DoubleArray(matrix.rows)
        val calls = mutableListOf<Pair<Int, Int>>()
        val selected = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override fun dotDense(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                dense: DoubleArray,
            ): Double {
                assertSame(matrix.rowIndices, indices)
                assertSame(matrix.values, values)
                assertSame(dense, denseOperand)
                assertEquals(indexOffset, valueOffset)
                calls += indexOffset to count
                return 0.0
            }
        }
        val kernels = SparseKernelAdapter("selected", ScalarVectorKernels, selected)

        for (column in 0 until matrix.cols) kernels.dotColumn(matrix, column, denseOperand)

        assertEquals(listOf(0 to 1, 1 to 2, 3 to 0, 3 to 0), calls)
    }

    @Test
    fun `stored value reductions call the selected dense kernels`() {
        for (count in listOf(0, 1, 3, 256)) {
            val x = SparseVector.of(1024, IntArray(count) { it * 4 }, DoubleArray(count) { 1.0 + it * 0.125 })
            val calls = mutableListOf<EngineImplementation>()
            fun record(operation: EngineImplementation, values: DoubleArray, offset: Int, length: Int, stride: Int) {
                assertSame(x.values, values)
                assertEquals(0, offset)
                assertEquals(count, length)
                assertEquals(1, stride)
                calls += operation
            }
            val selected = object : DenseVectorKernels by ScalarVectorKernels {
                override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                    record(EngineImplementation.Nrm2, v, vOff, len, vStride)
                    return 0.0
                }

                override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                    record(EngineImplementation.Asum, v, vOff, len, vStride)
                    return 0.0
                }
            }
            val kernels = SparseKernelAdapter("selected", selected, ScalarIndexedSparseKernels)

            kernels.nrm2(x)
            kernels.asum(x)

            assertEquals(listOf(EngineImplementation.Nrm2, EngineImplementation.Asum), calls, "count=$count")
        }
    }
}
