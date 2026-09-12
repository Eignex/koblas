package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

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
    fun `gather preserves slice boundaries and source values`() {
        for (engine in listOfNotNull(BuiltinEngines.c, BuiltinEngines.simd)) {
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
    fun `native gather retains ordered reads when buffers alias`() {
        val kernels = BuiltinEngines.c?.indexedSparseKernels ?: return
        val indices = IntArray(33) { it }
        val expected = DoubleArray(40) { it.toDouble() }
        val actual = expected.copyOf()
        ScalarIndexedSparseKernels.gather(indices, 0, expected, 3, indices.size, expected)

        kernels.gather(indices, 0, actual, 3, indices.size, actual)

        assertGatherAgreesWithReference(expected, actual, "aliased buffers")
    }

    private fun assertGatherAgreesWithReference(expected: DoubleArray, actual: DoubleArray, message: String) {
        assertContentEquals(expected, actual, message)
    }
}
