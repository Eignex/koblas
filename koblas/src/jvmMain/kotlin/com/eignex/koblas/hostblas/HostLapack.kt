package com.eignex.koblas.hostblas

import com.eignex.koblas.BackendNames
import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.Cblas.COL_MAJOR
import com.eignex.koblas.dense.F64HostLapackAdapter
import com.eignex.koblas.dense.F64PivotedQrDecomposition
import com.eignex.koblas.dense.F64QrDecomposition
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.rankOfPivotedR
import com.eignex.koblas.dispatchThresholds

/**
 * The host LAPACKE through `java.lang.foreign`. Every shared routine lives in [F64HostLapackAdapter]; this
 * supplies the JVM's entry points, its measured gates, and the one routine only this binding has.
 */
public class HostLapack internal constructor() : F64HostLapackAdapter(JvmLapackeCalls, JvmCblasCalls) {
    override val name: String get() = BackendNames.OPENBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** LAPACKE resolves separately from CBLAS, either inside OpenBLAS or as its own library. */
    override val isAvailable: Boolean get() = HostBlasCalls.lapackAvailable

    override val nativeTrsmMinRhs: Int get() = NATIVE_TRSM_MIN_RHS

    override val choleskyMin: Int get() = JVM_CHOLESKY_MIN

    override val spdInvertMin: Int get() = JVM_INVERT_SPD_MIN

    /**
     * A zeroed jpvt on input leaves every column free to move, and the output is one-based, hence the shift.
     * LAPACK reports no rank, so it comes from [rankOfPivotedR]. Only this binding has `dgeqp3`.
     */
    override fun qrPivoted(a: F64DenseMatrix, tolerance: Double, workspace: Workspace?): F64PivotedQrDecomposition {
        if (minOf(a.rows, a.cols) < dispatchThresholds.lapack) {
            return F64ReferenceLinearAlgebra.qrPivoted(a, tolerance, workspace)
        }
        val m = a.rows
        val n = a.cols
        val buf = a.data.copyOf()
        val tau = DoubleArray(minOf(m, n))
        val jpvt = IntArray(n) // zero: no column is pinned
        if (m > 0 && n > 0) {
            val info = HostBlasCalls.dgeqp3.invokeWithArguments(
                COL_MAJOR,
                m,
                n,
                HostBlasCalls.seg(buf),
                m,
                HostBlasCalls.seg(jpvt),
                HostBlasCalls.seg(tau),
            ) as Int
            check(info == 0) { "dgeqp3: illegal argument ${-info}" }
        }
        val pivots = IntArray(n) { jpvt[it] - 1 }
        val rank = rankOfPivotedR(buf, m, n, minOf(m, n), tolerance)
        return F64PivotedQrDecomposition(F64QrDecomposition(m, n, buf, tau), pivots, rank)
    }
}

/** Columns from which one blocked dtrsm beats a native call per column, measured on the JVM. */
private const val NATIVE_TRSM_MIN_RHS = 4

/** Where dpotrf takes over from the portable Cholesky. */
private const val JVM_CHOLESKY_MIN = 32

/** Where dpotri takes over from the portable SPD inverse. */
private const val JVM_INVERT_SPD_MIN = 16
