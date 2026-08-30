@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.dense

import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.*
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** The C level-1 kernels compiled into each Kotlin/Native host artifact. */
internal actual object F64PlatformKernels : F64Kernels {
    actual override val name: String get() = BackendNames.C

    override val isPortable: Boolean get() = true

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (len == 0) {
        0.0
    } else {
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_dot(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len == 0 || alpha == 0.0) return
        y.usePinned { yp ->
            x.usePinned { xp -> koblas_dense_axpy(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len) }
        }
    }

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (len == 0 || alpha == 1.0) return
        v.usePinned { vp -> koblas_dense_scale(vp.addressOf(0), vOff, alpha, len) }
    }

    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> koblas_dense_nrm2(vp.addressOf(0), vOff, len) }

    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> koblas_dense_asum(vp.addressOf(0), vOff, len) }

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (len == 0) return
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_swap(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    @Suppress("LongParameterList")
    override fun symvColumn(
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: Double,
        len: Int,
    ): Double {
        if (len == 0) return 0.0
        return a.usePinned { ap ->
            x.usePinned { xp ->
                y.usePinned { yp ->
                    koblas_dense_symv_column(
                        ap.addressOf(0),
                        aOff,
                        xp.addressOf(0),
                        xOff,
                        yp.addressOf(0),
                        yOff,
                        mult,
                        len,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList")
    override fun symvColumn4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: DoubleArray,
        out: DoubleArray,
        len: Int,
    ) {
        if (len == 0) {
            out.fill(0.0, 0, 4)
            return
        }
        a.usePinned { ap ->
            x.usePinned { xp ->
                y.usePinned { yp ->
                    mult.usePinned { mp ->
                        out.usePinned { op ->
                            koblas_dense_symv_column4(
                                ap.addressOf(0),
                                aOff,
                                stride,
                                xp.addressOf(0),
                                xOff,
                                yp.addressOf(0),
                                yOff,
                                mp.addressOf(0),
                                op.addressOf(0),
                                len,
                            )
                        }
                    }
                }
            }
        }
    }

    @Suppress("LongParameterList")
    actual override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        if (len == 0) {
            for (r in 0 until 4) out[outOff + r] = 0.0
            return
        }
        a.usePinned { ap ->
            b.usePinned { bp ->
                out.usePinned { op ->
                    koblas_dense_dot4(
                        ap.addressOf(0),
                        aOff,
                        stride,
                        bp.addressOf(0),
                        bOff,
                        len,
                        op.addressOf(0),
                        outOff,
                    )
                }
            }
        }
    }
}
