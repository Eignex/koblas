@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.dense.VectorKernels
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.usePinned

/**
 * The host OpenBLAS behind koblas's level-1 primitives, for the runs long enough to be worth a call.
 *
 * The [com.eignex.koblas.dense.LinearAlgebra] seam does not reach `dot`, `axpy` and `scale`: those compile per
 * target and, on native, are scalar loops measured 4x to 7x slower than the JVM's SIMD ones. Registering
 * this closes that gap for long vectors, which is where a simplex spends its level-1 time — the public
 * `dot`/`axpy`/`scale` and the eta-file ftran/btran both bottom out in these calls with no other seam
 * above them. koblas applies [com.eignex.koblas.DispatchThresholds.level1], so these methods only ever
 * see runs worth dispatching.
 *
 * Offsets are handled by pinning and taking the address of the element, so no repacking happens at the
 * boundary — the same zero-copy property the rest of this backend relies on.
 */
class CblasVectorKernels : VectorKernels {
    override val name: String get() = "cblas"

    /** Above the reference (0), matching the other cblas halves. */
    override val priority: Int get() = 90

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
