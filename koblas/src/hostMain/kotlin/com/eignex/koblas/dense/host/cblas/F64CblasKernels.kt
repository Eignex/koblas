@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.dense.host.blasOffset
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.portableRotmg
import com.eignex.koblas.toBlasParameters
import kotlinx.cinterop.*

/**
 * The host OpenBLAS behind koblas's level-1 primitives. koblas applies
 * the selected kernel backend, so these methods see every vector run,
 * but the [F64Kernels] contract makes a length of zero legal everywhere and an override of the
 * threshold routes those here. Each routine answers one itself: an empty run sits at the end of its array,
 * where there is no element to take an address of.
 */
public class F64CblasKernels internal constructor(
    private val loader: OpenBlasLoader,
    /** Policy for this level-1 backend instance. */
    public val config: HostBlasConfig,
) : F64Kernels {
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(OpenBlasLoader(config), config)
    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** The level-1 kernels come from CBLAS. */
    override val isAvailable: Boolean get() = loader.cblas != null

    private val f = requireNotNull(loader.cblas) {
        "OpenBLAS is not available on this host; koblas keeps its built-in level-1 kernels"
    }

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        if (len == 0) return 0.0
        return a.usePinned { ap ->
            b.usePinned { bp ->
                f.ddot(len, ap.addressOf(aOff), 1, bp.addressOf(bOff), 1)
            }
        }
    }

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                f.daxpy(len, alpha, xp.addressOf(xOff), 1, yp.addressOf(yOff), 1)
            }
        }
    }

    /** OpenBLAS has no plain sum, only `dasum` over absolute values, so this runs the compiled-in kernel. */
    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double = F64PlatformKernels.sum(v, vOff, len)

    /**
     * OpenBLAS has no fused sum of squared differences, and the only way to build one from the routines it
     * does have is `a.a - 2a.b + b.b`, whose cancellation ruins exactly the small values this is for. So
     * this one routine runs the compiled-in kernel instead of the host library.
     */
    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        F64PlatformKernels.ssqd(a, aOff, b, bOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (len == 0) return
        a.usePinned { ap -> b.usePinned { bp -> f.dswap(len, ap.addressOf(aOff), 1, bp.addressOf(bOff), 1) } }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (len == 0) return
        v.usePinned { vp -> f.dscal(len, alpha, vp.addressOf(vOff), 1) }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> f.dnrm2(len, vp.addressOf(vOff), 1) }

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> f.dasum(len, vp.addressOf(vOff), 1) }

    // OpenBLAS's rescaling state differs from Netlib's, so the public state contract stays portable.
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
        portableRotmg(d1, d2, x1, y1)

    @Suppress("LongParameterList")
    override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: F64ModifiedGivens,
    ) {
        if (len == 0 || transformation.flag == -2.0) return
        val parameters = transformation.toBlasParameters()
        x.usePinned { xp ->
            y.usePinned { yp ->
                parameters.usePinned { pp ->
                    f.drotm(
                        len,
                        xp.addressOf(blasOffset(xOff, xStride, len)),
                        xStride,
                        yp.addressOf(blasOffset(yOff, yStride, len)),
                        yStride,
                        pp.addressOf(0),
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (len == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                f.drot(len, xp.addressOf(xOff), 1, yp.addressOf(yOff), 1, c, s)
            }
        }
    }
}
