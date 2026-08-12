@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.VectorKernels
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.usePinned

/**
 * The host OpenBLAS behind koblas's level-1 primitives. koblas applies
 * [com.eignex.koblas.DispatchThresholds.level1], so these methods only see runs worth dispatching.
 */
public class CblasVectorKernels : VectorKernels {
    override val name: String get() = "cblas"

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    private val f = requireNotNull(OpenBlasLoader.cblas) {
        "OpenBLAS is not available on this host; koblas keeps its built-in level-1 kernels"
    }

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = a.usePinned { ap ->
        b.usePinned { bp ->
            f.ddot(len, ap.addressOf(aOff), 1, bp.addressOf(bOff), 1)
        }
    }

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        x.usePinned { xp ->
            y.usePinned { yp ->
                f.daxpy(len, alpha, xp.addressOf(xOff), 1, yp.addressOf(yOff), 1)
            }
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        v.usePinned { vp -> f.dscal(len, alpha, vp.addressOf(vOff), 1) }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        v.usePinned { vp -> f.dnrm2(len, vp.addressOf(vOff), 1) }

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        v.usePinned { vp -> f.dasum(len, vp.addressOf(vOff), 1) }
}
