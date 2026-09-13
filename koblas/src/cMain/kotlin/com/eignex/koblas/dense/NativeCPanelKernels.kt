@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.NativeCKernelBindings
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** Native compiled-C matrix-panel arithmetic with measured portable short-run fallbacks. */
internal class NativeCPanelKernels(private val bindings: NativeCKernelBindings) : DensePanelKernels {
    override fun dot4(
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
                    bindings.denseDot4(
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

    override fun axpy4(
        y: DoubleArray,
        yOff: Int,
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        c0: Double,
        c1: Double,
        c2: Double,
        c3: Double,
        len: Int,
    ) {
        if (len == 0) return
        y.usePinned { yp ->
            a.usePinned { ap ->
                bindings.denseAxpy4(
                    yp.addressOf(0), yOff, ap.addressOf(0), aOff, stride, c0, c1, c2, c3, len,
                )
            }
        }
    }

    override fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double {
        if (len == 0) return 0.0
        return y.usePinned { yp ->
            a.usePinned { ap ->
                x.usePinned { xp ->
                    bindings.denseDotAxpy(
                        yp.addressOf(0),
                        yOff,
                        alpha,
                        ap.addressOf(0),
                        aOff,
                        xp.addressOf(0),
                        xOff,
                        len,
                    )
                }
            }
        }
    }

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len == 0) return
        y.usePinned { yp ->
            x.usePinned { xp ->
                bindings.denseAxpyArithmetic(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len)
            }
        }
    }
}
