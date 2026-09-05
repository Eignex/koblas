@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.dense.host.CblasKernelCalls
import kotlinx.cinterop.*

/**
 * [CblasKernelCalls] over cinterop. Each array is pinned for the duration of its call and addressed at its
 * element offset, so the library reads and writes the Kotlin array in place rather than a copy.
 *
 * The symbols are resolved on use rather than on construction: a binding is constructible on a host without
 * OpenBLAS and reports that through `isAvailable`, which it could not do if building one raised.
 */
@Suppress("LongParameterList") // the CBLAS signatures
internal class NativeCblasKernelCalls(private val loader: OpenBlasLoader) : CblasKernelCalls {
    private val f: CblasFunctions
        get() = checkNotNull(loader.cblas) {
            "OpenBLAS is not available on this host; koblas keeps its built-in level-1 kernels"
        }

    override fun ddot(n: Int, x: DoubleArray, xOffset: Int, incx: Int, y: DoubleArray, yOffset: Int, incy: Int) =
        x.usePinned { xp -> y.usePinned { yp -> f.ddot(n, xp.addressOf(xOffset), incx, yp.addressOf(yOffset), incy) } }

    override fun daxpy(
        n: Int,
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
    ) {
        x.usePinned { xp ->
            y.usePinned { yp -> f.daxpy(n, alpha, xp.addressOf(xOffset), incx, yp.addressOf(yOffset), incy) }
        }
    }

    override fun dswap(n: Int, x: DoubleArray, xOffset: Int, incx: Int, y: DoubleArray, yOffset: Int, incy: Int) {
        x.usePinned { xp -> y.usePinned { yp -> f.dswap(n, xp.addressOf(xOffset), incx, yp.addressOf(yOffset), incy) } }
    }

    override fun dscal(n: Int, alpha: Double, x: DoubleArray, xOffset: Int, incx: Int) {
        x.usePinned { xp -> f.dscal(n, alpha, xp.addressOf(xOffset), incx) }
    }

    override fun dnrm2(n: Int, x: DoubleArray, xOffset: Int, incx: Int) =
        x.usePinned { xp -> f.dnrm2(n, xp.addressOf(xOffset), incx) }

    override fun dasum(n: Int, x: DoubleArray, xOffset: Int, incx: Int) =
        x.usePinned { xp -> f.dasum(n, xp.addressOf(xOffset), incx) }

    override fun drotm(
        n: Int,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
        parameters: DoubleArray,
    ) {
        x.usePinned { xp ->
            y.usePinned { yp ->
                parameters.usePinned { pp ->
                    f.drotm(n, xp.addressOf(xOffset), incx, yp.addressOf(yOffset), incy, pp.addressOf(0))
                }
            }
        }
    }

    override fun drot(
        n: Int,
        x: DoubleArray,
        xOffset: Int,
        incx: Int,
        y: DoubleArray,
        yOffset: Int,
        incy: Int,
        c: Double,
        s: Double,
    ) {
        x.usePinned { xp ->
            y.usePinned { yp -> f.drot(n, xp.addressOf(xOffset), incx, yp.addressOf(yOffset), incy, c, s) }
        }
    }
}
