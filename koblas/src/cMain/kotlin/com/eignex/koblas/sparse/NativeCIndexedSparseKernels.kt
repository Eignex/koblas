@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.sparse

import com.eignex.koblas.internal.kernels.*
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** C indexed data movement, with ordered and zero-evaluating matrix arithmetic retained by the scalar delegate. */
internal object NativeCIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    override fun dotDense(indices: IntArray, values: DoubleArray, dense: DoubleArray): Double {
        if (values.isEmpty()) return 0.0
        return indices.usePinned { ip ->
            values.usePinned { vp ->
                dense.usePinned { dp ->
                    koblas_sparse_dot_dense(
                        ip.addressOf(0),
                        vp.addressOf(0),
                        values.size,
                        dp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun dotSparse(
        xIndices: IntArray,
        xValues: DoubleArray,
        yIndices: IntArray,
        yValues: DoubleArray,
    ): Double = dotSparse(xIndices, xValues, 0, xValues.size, yIndices, yValues, 0, yValues.size)

    override fun dotSparse(
        xIndices: IntArray,
        xValues: DoubleArray,
        xFromIndex: Int,
        xToIndex: Int,
        yIndices: IntArray,
        yValues: DoubleArray,
        yFromIndex: Int,
        yToIndex: Int,
    ): Double {
        if (xFromIndex == xToIndex || yFromIndex == yToIndex) return 0.0
        return xIndices.usePinned { xip ->
            xValues.usePinned { xvp ->
                yIndices.usePinned { yip ->
                    yValues.usePinned { yvp ->
                        koblas_sparse_dot_sparse(
                            xip.addressOf(xFromIndex),
                            xvp.addressOf(xFromIndex),
                            xToIndex - xFromIndex,
                            yip.addressOf(yFromIndex),
                            yvp.addressOf(yFromIndex),
                            yToIndex - yFromIndex,
                        )
                    }
                }
            }
        }
    }

    override fun axpy(indices: IntArray, values: DoubleArray, alpha: Double, destination: DoubleArray) {
        if (values.isEmpty()) return
        indices.usePinned { ip ->
            values.usePinned { vp ->
                destination.usePinned { dp ->
                    koblas_sparse_axpy(
                        ip.addressOf(0),
                        vp.addressOf(0),
                        values.size,
                        alpha,
                        dp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun scatter(indices: IntArray, values: DoubleArray, destination: DoubleArray) =
        scatter(indices, values, 0, values.size, destination)

    override fun scatter(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        destination: DoubleArray,
    ) {
        if (fromIndex == toIndex) return
        indices.usePinned { ip ->
            values.usePinned { vp ->
                destination.usePinned { dp ->
                    koblas_sparse_scatter(
                        ip.addressOf(fromIndex),
                        vp.addressOf(fromIndex),
                        toIndex - fromIndex,
                        dp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun gather(indices: IntArray, values: DoubleArray, source: DoubleArray) =
        gather(indices, values, 0, values.size, source)

    override fun gather(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, source: DoubleArray) {
        if (fromIndex == toIndex) return
        indices.usePinned { ip ->
            values.usePinned { vp ->
                source.usePinned { sp ->
                    koblas_sparse_gather(
                        ip.addressOf(fromIndex),
                        vp.addressOf(fromIndex),
                        toIndex - fromIndex,
                        sp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun gatherZero(indices: IntArray, values: DoubleArray, source: DoubleArray) =
        gatherZero(indices, values, 0, values.size, source)

    override fun gatherZero(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        source: DoubleArray,
    ) {
        if (fromIndex == toIndex) return
        indices.usePinned { ip ->
            values.usePinned { vp ->
                source.usePinned { sp ->
                    koblas_sparse_gather_zero(
                        ip.addressOf(fromIndex),
                        vp.addressOf(fromIndex),
                        toIndex - fromIndex,
                        sp.addressOf(0),
                    )
                }
            }
        }
    }
}
