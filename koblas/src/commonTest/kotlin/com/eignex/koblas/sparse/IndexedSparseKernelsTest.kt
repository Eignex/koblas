package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.SparseVector
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class IndexedSparseKernelsTest {
    @Test
    fun `whole vector gather calls the selected slice implementation`() {
        val indices = intArrayOf(1, 3)
        val values = DoubleArray(2)
        val source = DoubleArray(4)
        var calls = 0
        val kernels = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override fun gather(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                source: DoubleArray,
            ) {
                assertEquals(0, indexOffset)
                assertEquals(0, valueOffset)
                assertEquals(2, count)
                calls++
            }
        }

        kernels.gather(indices, values, source)

        assertEquals(1, calls)
    }

    @Test
    fun `whole vector dense dot calls the selected slice implementation`() {
        val indices = intArrayOf(1, 3)
        val values = DoubleArray(2)
        val source = DoubleArray(4)
        var calls = 0
        val kernels = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override fun dotDense(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                dense: DoubleArray,
            ): Double {
                assertEquals(0, indexOffset)
                assertEquals(0, valueOffset)
                assertEquals(2, count)
                calls++
                return 0.0
            }
        }

        kernels.dotDense(indices, values, source)

        assertEquals(1, calls)
    }

    @Test
    fun `whole vector axpy calls the selected slice implementation`() {
        val indices = intArrayOf(1, 3)
        val values = DoubleArray(2)
        val source = DoubleArray(4)
        var calls = 0
        val kernels = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override fun axpy(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                alpha: Double,
                destination: DoubleArray,
            ) {
                assertEquals(0, indexOffset)
                assertEquals(0, valueOffset)
                assertEquals(2, count)
                assertEquals(0.75, alpha)
                calls++
            }
        }

        kernels.axpy(indices, values, 0.75, source)

        assertEquals(1, calls)
    }

    @Test
    fun `whole vector scatter calls the selected slice implementation`() {
        val indices = intArrayOf(1, 3)
        val values = DoubleArray(2)
        val source = DoubleArray(4)
        var calls = 0
        val kernels = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override fun scatter(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                destination: DoubleArray,
            ) {
                assertEquals(0, indexOffset)
                assertEquals(0, valueOffset)
                assertEquals(2, count)
                calls++
            }
        }

        kernels.scatter(indices, values, source)

        assertEquals(1, calls)
    }

    @Test
    fun `whole vector gather zero calls the selected slice implementation`() {
        val indices = intArrayOf(1, 3)
        val values = DoubleArray(2)
        val source = DoubleArray(4)
        var calls = 0
        val kernels = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override fun gatherZero(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                source: DoubleArray,
            ) {
                assertEquals(0, indexOffset)
                assertEquals(0, valueOffset)
                assertEquals(2, count)
                calls++
            }
        }

        kernels.gatherZero(indices, values, source)

        assertEquals(1, calls)
    }

    @Test
    fun `whole vector sparse dot calls the selected slice implementation`() {
        val xIndices = intArrayOf(1, 3)
        val xValues = DoubleArray(2)
        val yIndices = intArrayOf(0, 1, 3)
        val yValues = DoubleArray(3)
        var calls = 0
        val kernels = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override fun dotSparse(
                xIndices: IntArray,
                xIndexOffset: Int,
                xValues: DoubleArray,
                xValueOffset: Int,
                xCount: Int,
                yIndices: IntArray,
                yIndexOffset: Int,
                yValues: DoubleArray,
                yValueOffset: Int,
                yCount: Int,
            ): Double {
                assertEquals(0, xIndexOffset)
                assertEquals(0, xValueOffset)
                assertEquals(2, xCount)
                assertEquals(0, yIndexOffset)
                assertEquals(0, yValueOffset)
                assertEquals(3, yCount)
                calls++
                return 0.0
            }
        }

        kernels.dotSparse(xIndices, xValues, yIndices, yValues)

        assertEquals(1, calls)
    }

    @Test
    fun `gather preserves slice boundaries and source values`() {
        for (engine in listOfNotNull(BuiltinEngines.simd)) {
            for (count in listOf(0, 1, 3, 4, 7, 15, 16, 17, 31, 32, 33, 40, 127, 128, 129, 1024)) {
                for (offset in listOf(0, 3)) {
                    val indices = IntArray(offset + count + 2) { if (it < offset) -1 else 2 * (it - offset) }
                    val source = DoubleArray(count * 2 + 5) { if (it % 7 == 0) Double.NaN else it * 0.125 }
                    val expected = DoubleArray(offset + count + 3) { -0.0 }
                    val actual = expected.copyOf()
                    val before = source.copyOf()
                    ScalarIndexedSparseKernels.gather(indices, offset, expected, offset, count, source)

                    engine.indexedSparseKernels.gather(indices, offset, actual, offset, count, source)

                    assertGatherAgreesWithReference(expected, actual, "count=$count offset=$offset")
                    assertContentEquals(before, source)
                }
            }
        }
    }

    @Test
    fun `gather rejects a source that is its own destination`() {
        val x = SparseVector.of(4, intArrayOf(0, 1, 2, 3), doubleArrayOf(0.0, 1.0, 2.0, 3.0))

        assertFailsWith<IllegalArgumentException> { koblas.sparseKernels.gather(x, x.values) }
        assertFailsWith<IllegalArgumentException> { koblas.sparseKernels.gatherZero(x, x.values) }
        assertContentEquals(doubleArrayOf(0.0, 1.0, 2.0, 3.0), x.values)
    }

    private fun assertGatherAgreesWithReference(expected: DoubleArray, actual: DoubleArray, message: String) {
        assertContentEquals(expected, actual, message)
    }
}
