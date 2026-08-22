@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.F64PivotedQrDecomposition
import com.eignex.koblas.dense.F64QrDecomposition
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.host.F64HostLapackAdapter
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.dense.rankOfPivotedR
import com.eignex.koblas.dense.requireRankTolerance
import com.eignex.koblas.f64DispatchThresholds
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

    /**
     * A zeroed jpvt on input leaves every column free to move, and the output is one-based, hence the shift.
     * LAPACK reports no rank, so it comes from [rankOfPivotedR].
     *
     * `dgeqp3` is an optional binding, so a LAPACKE without it takes the portable path here rather than
     * costing the host its whole LAPACK half.
     */
    override fun qrPivoted(a: F64DenseMatrix, tolerance: Double, workspace: Workspace?): F64PivotedQrDecomposition {
        requireRankTolerance(tolerance)
        val dgeqp3 = f.dgeqp3
        // The floor of one is what keeps an empty dimension off the native call, rather than PIVOTED_QR_MIN
        // happening to sit above zero: dgeqp3 has nothing to factor then, and the untouched jpvt would read
        // back as a column permutation of -1. Same floor as the JVM binding.
        if (dgeqp3 == null || minOf(a.rows, a.cols) < maxOf(1, f64DispatchThresholds.lapack, PIVOTED_QR_MIN)) {
            return F64ReferenceLinearAlgebra.qrPivoted(a, tolerance, workspace)
        }
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        val jpvt = IntArray(n) // zero: no column is pinned
        val info = buf.usePinned { bp ->
            jpvt.usePinned { jp ->
                tau.usePinned { tp ->
                    dgeqp3(COL_MAJOR, m, n, bp.addressOf(0), m, jp.addressOf(0), tp.addressOf(0))
                }
            }
        }
        check(info == 0) { "dgeqp3: illegal argument ${-info}" }
        val pivots = IntArray(n) { jpvt[it] - 1 }
        val rank = rankOfPivotedR(buf, m, n, minOf(m, n), tolerance)
        return F64PivotedQrDecomposition(F64QrDecomposition(m, n, buf, tau), pivots, rank)
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
