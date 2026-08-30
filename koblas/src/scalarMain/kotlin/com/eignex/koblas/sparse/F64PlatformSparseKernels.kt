@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.requireShape
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** The C sparse level-1 kernels compiled into every Kotlin/Native target. */
internal actual object F64PlatformSparseKernels : F64SparseKernels {
    actual override val name: String get() = BackendNames.C_SPARSE

    override val isPortable: Boolean get() = true

    actual override fun dot(x: F64SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        if (x.values.isEmpty()) return 0.0
        return x.indices.usePinned { ip ->
            x.values.usePinned { vp ->
                y.usePinned { yp ->
                    koblas_sparse_dot_dense(ip.addressOf(0), vp.addressOf(0), x.values.size, yp.addressOf(0))
                }
            }
        }
    }

    actual override fun dot(x: F64SparseVector, y: F64SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        if (x.values.isEmpty() || y.values.isEmpty()) return 0.0
        return x.indices.usePinned { xip ->
            x.values.usePinned { xvp ->
                y.indices.usePinned { yip ->
                    y.values.usePinned { yvp ->
                        koblas_sparse_dot_sparse(
                            xip.addressOf(0),
                            xvp.addressOf(0),
                            x.values.size,
                            yip.addressOf(0),
                            yvp.addressOf(0),
                            y.values.size,
                        )
                    }
                }
            }
        }
    }

    actual override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector) {
        requireShape(y.size == x.size) { "axpy: sizes differ, ${y.size} vs ${x.size}" }
        if (x.values.isEmpty() || alpha == 0.0) return
        x.indices.usePinned { ip ->
            x.values.usePinned { vp ->
                y.usePinned { yp ->
                    koblas_sparse_axpy(ip.addressOf(0), vp.addressOf(0), x.values.size, alpha, yp.addressOf(0))
                }
            }
        }
    }

    actual override fun scatter(x: F64SparseVector, out: DoubleArray) {
        requireShape(out.size == x.size) { "scatter: sizes differ, ${out.size} vs ${x.size}" }
        if (x.values.isEmpty()) return
        x.indices.usePinned { ip ->
            x.values.usePinned { vp ->
                out.usePinned { op ->
                    koblas_sparse_scatter(ip.addressOf(0), vp.addressOf(0), x.values.size, op.addressOf(0))
                }
            }
        }
    }

    actual override fun gather(x: F64SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gather: sizes differ, ${from.size} vs ${x.size}" }
        if (x.values.isEmpty()) return
        x.indices.usePinned { ip ->
            x.values.usePinned { vp ->
                from.usePinned { fp ->
                    koblas_sparse_gather(ip.addressOf(0), vp.addressOf(0), x.values.size, fp.addressOf(0))
                }
            }
        }
    }

    actual override fun gatherZero(x: F64SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gatherZero: sizes differ, ${from.size} vs ${x.size}" }
        if (x.values.isEmpty()) return
        x.indices.usePinned { ip ->
            x.values.usePinned { vp ->
                from.usePinned { fp ->
                    koblas_sparse_gather_zero(ip.addressOf(0), vp.addressOf(0), x.values.size, fp.addressOf(0))
                }
            }
        }
    }

    actual override fun nrm2(x: F64SparseVector): Double = F64PlatformKernels.nrm2(x.values, 0, x.values.size)

    actual override fun asum(x: F64SparseVector): Double = F64PlatformKernels.asum(x.values, 0, x.values.size)
}
