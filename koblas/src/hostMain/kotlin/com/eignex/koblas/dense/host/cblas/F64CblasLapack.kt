@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64HostLapackAdapter
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.internal.backend.BackendNames
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.usePinned

/**
 * The host LAPACKE through cinterop. Every shared routine lives in [F64HostLapackAdapter]; this supplies the
 * native entry points and the backend's identity. The gates keep their defaults, so native dispatches at any
 * size.
 */
internal class F64CblasLapack(private val f: LapackeFunctions, blas: CblasFunctions) :
    F64HostLapackAdapter(NativeLapackeCalls(f), NativeCblasCalls(blas)) {
    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** LAPACKE calls into CBLAS, so this half needs both. */
    override val isAvailable: Boolean get() = OpenBlasLoader.lapacke != null && OpenBlasLoader.cblas != null

    override val pivotedQrMin: Int get() = PIVOTED_QR_MIN

    override fun dgeqp3(m: Int, n: Int, a: DoubleArray, jpvt: IntArray, tau: DoubleArray): Int? {
        val dgeqp3 = f.dgeqp3 ?: return null
        return a.usePinned { ap ->
            jpvt.usePinned { jp ->
                tau.usePinned { tp -> dgeqp3(COL_MAJOR, m, n, ap.addressOf(0), m, jp.addressOf(0), tp.addressOf(0)) }
            }
        }
    }
}

/**
 * Dimension from which `dgeqp3` beats the portable pivoted QR on native.
 *
 * The native LAPACK threshold is zero, which suits the routines it was chosen for but not this one, since a
 * pivoted QR does more setup per call. Measured with the `pivotedQrGate` suite: at 4 the host is 1.3x slower
 * square and 1.4x tall, at 8 the two are within each other's error, and from 12 the host wins by 1.4x and up.
 */
private const val PIVOTED_QR_MIN = 8
