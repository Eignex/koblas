package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

class SparseKernelAdapterTest {
    @Test
    fun `stored value reductions call the selected dense kernels`() {
        for (count in listOf(0, 1, 3, 256)) {
            val x = SparseVector.of(1024, IntArray(count) { it * 4 }, DoubleArray(count) { 1.0 + it * 0.125 })
            val calls = mutableListOf<DenseOperation>()
            fun record(operation: DenseOperation, values: DoubleArray, offset: Int, length: Int, stride: Int) {
                assertSame(x.values, values)
                assertEquals(0, offset)
                assertEquals(count, length)
                assertEquals(1, stride)
                calls += operation
            }
            val selected = object : DenseVectorKernels by ScalarVectorKernels {
                override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                    record(DenseOperation.Nrm2, v, vOff, len, vStride)
                    return 0.0
                }

                override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                    record(DenseOperation.Asum, v, vOff, len, vStride)
                    return 0.0
                }
            }
            val kernels = SparseKernelAdapter("selected", selected, ScalarIndexedSparseKernels)

            kernels.nrm2(x)
            kernels.asum(x)

            assertEquals(listOf(DenseOperation.Nrm2, DenseOperation.Asum), calls, "count=$count")
        }
    }
}
