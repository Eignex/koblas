@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.sparse

import com.eignex.koblas.internal.kernels.NativeCKernelBindings
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** Kotlin/Native bindings to the indexed C leaves, with scalar fallbacks for short slices. */
internal class NativeCIndexedSparseKernels(private val bindings: NativeCKernelBindings) :
    IndexedSparseKernels by ScalarIndexedSparseKernels {
    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double {
        if (count < SparseTuning.nativeIndexedCrossover) {
            return ScalarIndexedSparseKernels.dotDense(indices, indexOffset, values, valueOffset, count, dense)
        }
        return indices.usePinned { ip ->
            values.usePinned { vp ->
                dense.usePinned { dp ->
                    bindings.sparseDotDense(
                        ip.addressOf(0),
                        indexOffset,
                        vp.addressOf(0),
                        valueOffset,
                        count,
                        dp.addressOf(0),
                    )
                }
            }
        }
    }

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
        if (xCount == 0 || yCount == 0) return 0.0
        return xIndices.usePinned { xip ->
            xValues.usePinned { xvp ->
                yIndices.usePinned { yip ->
                    yValues.usePinned { yvp ->
                        bindings.sparseDotSparse(
                            xip.addressOf(xIndexOffset),
                            xvp.addressOf(xValueOffset),
                            xCount,
                            yip.addressOf(yIndexOffset),
                            yvp.addressOf(yValueOffset),
                            yCount,
                        )
                    }
                }
            }
        }
    }

    override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        if (count < SparseTuning.nativeIndexedCrossover) {
            ScalarIndexedSparseKernels.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
            return
        }
        indices.usePinned { ip ->
            values.usePinned { vp ->
                destination.usePinned { dp ->
                    bindings.sparseAxpy(
                        ip.addressOf(0),
                        indexOffset,
                        vp.addressOf(0),
                        valueOffset,
                        count,
                        alpha,
                        dp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        if (count < SparseTuning.nativeIndexedCrossover) {
            ScalarIndexedSparseKernels.scatter(indices, indexOffset, values, valueOffset, count, destination)
            return
        }
        indices.usePinned { ip ->
            values.usePinned { vp ->
                destination.usePinned { dp ->
                    bindings.sparseScatter(
                        ip.addressOf(0),
                        indexOffset,
                        vp.addressOf(0),
                        valueOffset,
                        count,
                        dp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        if (count < SparseTuning.nativeIndexedCrossover) {
            ScalarIndexedSparseKernels.gather(indices, indexOffset, values, valueOffset, count, source)
            return
        }
        indices.usePinned { ip ->
            values.usePinned { vp ->
                source.usePinned { sp ->
                    bindings.sparseGather(
                        ip.addressOf(indexOffset),
                        vp.addressOf(valueOffset),
                        count,
                        sp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        if (count == 0) return
        indices.usePinned { ip ->
            values.usePinned { vp ->
                source.usePinned { sp ->
                    bindings.sparseGatherZero(
                        ip.addressOf(indexOffset),
                        vp.addressOf(valueOffset),
                        count,
                        sp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
        if (count < SparseTuning.nativeIndexedCrossover) {
            return ScalarIndexedSparseKernels.nrm2(indices, indexOffset, count, values)
        }
        return indices.usePinned { ip ->
            values.usePinned { vp -> bindings.sparseNrm2(ip.addressOf(0), indexOffset, count, vp.addressOf(0)) }
        }
    }
}
