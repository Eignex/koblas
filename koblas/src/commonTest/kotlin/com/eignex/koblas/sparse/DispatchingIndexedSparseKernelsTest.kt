package com.eignex.koblas.sparse

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class DispatchingIndexedSparseKernelsTest {
    @Test
    fun `routes identify the leaf that executes each indexed operation`() {
        val operations = SparseOperation.entries.filter { it != SparseOperation.Nrm2 && it != SparseOperation.Asum }
        for (operation in operations) {
            for (count in listOf(0, 1, 15, 16, 17)) {
                for (offset in listOf(0, 2)) {
                    val calls = mutableListOf<Call>()
                    val fallback = RecordingLeaf("fallback", calls)
                    val wide = RecordingLeaf("wide", calls)
                    val kernels = object : DispatchingIndexedSparseKernels() {
                        override val name = "dispatch"
                        override fun select(operation: SparseOperation, count: Int): IndexedSparseKernels =
                            if ((count >= 16) == (operation != SparseOperation.DotSparse)) wide else fallback
                    }
                    val indices = IntArray(count + offset) { it }
                    val values = DoubleArray(count + offset)
                    val dense = DoubleArray(count + offset + 1)
                    val yIndices = IntArray(count + offset + 1) { it }
                    val yValues = DoubleArray(count + offset + 1)
                    val expectedArguments: List<Any> = when (operation) {
                        SparseOperation.DotSparse -> listOf(
                            indices, offset, values, offset, count, yIndices, offset, yValues, offset, count + 1,
                        )

                        SparseOperation.IndexedNrm2 -> listOf(indices, offset, count, dense)

                        SparseOperation.Axpy -> listOf(indices, offset, values, offset, count, 0.75, dense)

                        else -> listOf(indices, offset, values, offset, count, dense)
                    }
                    val route = kernels.implementationFor(operation, count)

                    when (operation) {
                        SparseOperation.DotDense -> if (offset == 0) {
                            kernels.dotDense(indices, values, dense)
                        } else {
                            kernels.dotDense(indices, offset, values, offset, count, dense)
                        }

                        SparseOperation.DotSparse -> if (offset == 0) {
                            kernels.dotSparse(indices, values, yIndices, yValues)
                        } else {
                            kernels.dotSparse(
                                indices, offset, values, offset, count, yIndices, offset, yValues, offset,
                                count + 1,
                            )
                        }

                        SparseOperation.Axpy -> if (offset == 0) {
                            kernels.axpy(indices, values, 0.75, dense)
                        } else {
                            kernels.axpy(indices, offset, values, offset, count, 0.75, dense)
                        }

                        SparseOperation.Scatter -> if (offset == 0) {
                            kernels.scatter(indices, values, dense)
                        } else {
                            kernels.scatter(indices, offset, values, offset, count, dense)
                        }

                        SparseOperation.Gather -> if (offset == 0) {
                            kernels.gather(indices, values, dense)
                        } else {
                            kernels.gather(indices, offset, values, offset, count, dense)
                        }

                        SparseOperation.GatherZero -> if (offset == 0) {
                            kernels.gatherZero(indices, values, dense)
                        } else {
                            kernels.gatherZero(indices, offset, values, offset, count, dense)
                        }

                        SparseOperation.IndexedNrm2 -> kernels.nrm2(indices, offset, count, dense)

                        SparseOperation.Nrm2, SparseOperation.Asum -> error("contiguous reductions use dense kernels")
                    }

                    val call = calls.single()
                    assertEquals(operation, call.operation)
                    assertEquals(expectedArguments, call.arguments)
                    assertEquals(call.leaf, route, "$operation at $count with offset $offset")
                }
            }
        }
    }

    @Test
    fun `value dependent leaf attribution remains unknown`() {
        val leaf = object : IndexedSparseKernels by ScalarIndexedSparseKernels {
            override fun implementationFor(operation: SparseOperation, count: Int): String? = null
        }
        val kernels = object : DispatchingIndexedSparseKernels() {
            override val name = "dispatch"
            override fun select(operation: SparseOperation, count: Int): IndexedSparseKernels = leaf
        }

        val route = kernels.implementationFor(SparseOperation.IndexedNrm2, 16)

        assertNull(route)
    }

    private data class Call(val leaf: String, val operation: SparseOperation, val arguments: List<Any>)

    private class RecordingLeaf(override val name: String, private val calls: MutableList<Call>) :
        IndexedSparseKernels {
        private fun record(operation: SparseOperation, vararg arguments: Any) {
            calls += Call(name, operation, arguments.toList())
        }

        @Suppress("LongParameterList")
        override fun dotDense(
            indices: IntArray,
            indexOffset: Int,
            values: DoubleArray,
            valueOffset: Int,
            count: Int,
            dense: DoubleArray,
        ): Double {
            record(SparseOperation.DotDense, indices, indexOffset, values, valueOffset, count, dense)
            return 0.0
        }

        @Suppress("LongParameterList")
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
            record(
                SparseOperation.DotSparse,
                xIndices, xIndexOffset, xValues, xValueOffset, xCount,
                yIndices, yIndexOffset, yValues, yValueOffset, yCount,
            )
            return 0.0
        }

        @Suppress("LongParameterList")
        override fun axpy(
            indices: IntArray,
            indexOffset: Int,
            values: DoubleArray,
            valueOffset: Int,
            count: Int,
            alpha: Double,
            destination: DoubleArray,
        ) {
            record(SparseOperation.Axpy, indices, indexOffset, values, valueOffset, count, alpha, destination)
        }

        @Suppress("LongParameterList")
        override fun scatter(
            indices: IntArray,
            indexOffset: Int,
            values: DoubleArray,
            valueOffset: Int,
            count: Int,
            destination: DoubleArray,
        ) {
            record(SparseOperation.Scatter, indices, indexOffset, values, valueOffset, count, destination)
        }

        @Suppress("LongParameterList")
        override fun gather(
            indices: IntArray,
            indexOffset: Int,
            values: DoubleArray,
            valueOffset: Int,
            count: Int,
            source: DoubleArray,
        ) {
            record(SparseOperation.Gather, indices, indexOffset, values, valueOffset, count, source)
        }

        @Suppress("LongParameterList")
        override fun gatherZero(
            indices: IntArray,
            indexOffset: Int,
            values: DoubleArray,
            valueOffset: Int,
            count: Int,
            source: DoubleArray,
        ) {
            record(SparseOperation.GatherZero, indices, indexOffset, values, valueOffset, count, source)
        }

        override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
            record(SparseOperation.IndexedNrm2, indices, indexOffset, count, values)
            return 0.0
        }
    }
}
