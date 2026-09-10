@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
@file:Suppress("MatchingDeclarationName") // actual family selection shares this source set boundary

package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.NativeCPanelKernels
import com.eignex.koblas.dense.PlatformVectorKernels
import com.eignex.koblas.internal.configuration.ImplementationNames
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.requireShape
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** C indexed data movement, with ordered and zero-evaluating matrix arithmetic retained by the scalar delegate. */
internal object NativeCIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
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

internal object NativeCSparseKernels : SparseKernels {
    override val name: String get() = ImplementationNames.C_SPARSE

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        if (x.values.isEmpty()) return 0.0
        return x.indices.usePinned { ip ->
            x.values.usePinned { vp ->
                y.usePinned { yp ->
                    koblas_sparse_dot_dense(
                        ip.addressOf(0),
                        vp.addressOf(0),
                        x.values.size,
                        yp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun dot(x: SparseVector, y: SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return NativeCIndexedSparseKernels.dotSparse(
            x.indices,
            x.values,
            0,
            x.values.size,
            y.indices,
            y.values,
            0,
            y.values.size,
        )
    }

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0 || x.values.isEmpty()) return
        x.indices.usePinned { ip ->
            x.values.usePinned { vp ->
                y.usePinned { yp ->
                    koblas_sparse_axpy(
                        ip.addressOf(0),
                        vp.addressOf(0),
                        x.values.size,
                        alpha,
                        yp.addressOf(0),
                    )
                }
            }
        }
    }

    override fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        NativeCIndexedSparseKernels.scatter(x.indices, x.values, 0, x.values.size, out)
    }

    override fun gather(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
        NativeCIndexedSparseKernels.gather(x.indices, x.values, 0, x.values.size, from)
    }

    override fun gatherZero(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
        NativeCIndexedSparseKernels.gatherZero(x.indices, x.values, 0, x.values.size, from)
    }

    override fun nrm2(x: SparseVector): Double = PlatformVectorKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: SparseVector): Double = PlatformVectorKernels.asum(x.values, 0, x.values.size)
}

internal val nativeCSparseKernelFamilies: SparseKernelFamilies = sparseKernelFamilies(
    NativeCSparseKernels,
    NativeCIndexedSparseKernels,
    PlatformVectorKernels,
    NativeCPanelKernels,
)

internal actual val platformSparseKernelFamilies: SparseKernelFamilies = nativeCSparseKernelFamilies
