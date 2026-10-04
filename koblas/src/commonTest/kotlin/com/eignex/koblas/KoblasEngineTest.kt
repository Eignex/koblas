package com.eignex.koblas

import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.IndexedSparseKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseOperation
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

class KoblasEngineTest {
    @Test
    fun `sparse convenience operations execute the engine selected kernels`() {
        val x = SparseVector.of(4, intArrayOf(1, 3), doubleArrayOf(2.0, 4.0))
        val calls = mutableListOf<String>()
        val dense = object : DenseVectorKernels by ScalarVectorKernels {
            override val name = "selected-dense"
            override fun implementationFor(operation: DenseOperation, length: Int, contiguous: Boolean): String = name
            override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertSame(x.values, v)
                calls += name
                return 0.0
            }
        }
        val indexed = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override val name = "selected-indexed"
            override fun implementationFor(operation: SparseOperation, count: Int): String = name
            override fun dotDense(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                dense: DoubleArray,
            ): Double {
                assertSame(x.indices, indices)
                assertSame(x.values, values)
                calls += name
                return 0.0
            }
        }
        val engine = KoblasEngine(dense, indexed, null)

        engine.sparseKernels.nrm2(x)
        engine.sparseKernels.dot(x, DoubleArray(x.size))

        assertEquals(
            listOf(
                engine.routeOf(SparseOperation.Nrm2, x.values.size).implementation,
                engine.routeOf(SparseOperation.DotDense, x.values.size).implementation,
            ),
            calls,
        )
        assertEquals(listOf(dense.name, indexed.name), calls)
    }
}
