@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.Level1Kernels
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.usePinned

/**
 * The host OpenBLAS behind koblas's level-1 primitives, for the runs long enough to be worth a call.
 *
 * The [com.eignex.koblas.LinearAlgebra] seam does not reach `dot`, `axpy` and `scale`: those compile per
 * target and, on native, are scalar loops measured 4x to 7x slower than the JVM's SIMD ones. Installing
 * this closes that gap for long vectors, which is where a simplex or a Cholesky update spends its
 * level-1 time. koblas applies the length threshold, so these methods only ever see runs worth
 * dispatching.
 *
 * Offsets are handled by pinning and taking the address of the element, so no repacking happens at the
 * boundary — the same zero-copy property the rest of this backend relies on.
 */
class CblasLevel1Kernels : Level1Kernels {

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
}
