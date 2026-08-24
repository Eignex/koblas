@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.backend.hostBlasDispatchThresholds
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.usePinned

/**
 * The host OpenBLAS behind koblas's level-1 primitives. koblas applies
 * the level-1 dispatch threshold, so these methods normally see only runs worth dispatching,
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
    override val minDispatchLength: Int get() = hostBlasDispatchThresholds(config).level1

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

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (len == 0) return
        v.usePinned { vp -> f.dscal(len, alpha, vp.addressOf(vOff), 1) }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> f.dnrm2(len, vp.addressOf(vOff), 1) }

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> f.dasum(len, vp.addressOf(vOff), 1) }
}
