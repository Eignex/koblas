package com.eignex.koblas.hostblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64HostLapackAdapter
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.dense.host.jvm.HostBlasCalls
import com.eignex.koblas.dense.host.jvm.JvmCblasCalls
import com.eignex.koblas.dense.host.jvm.JvmLapackeCalls
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The host LAPACKE through `java.lang.foreign`. Every shared routine lives in [F64HostLapackAdapter]; this
 * supplies the JVM's entry points and measured gates.
 */
public class HostLapack : F64HostLapackAdapter(JvmLapackeCalls, JvmCblasCalls) {
    override val name: String get() = BackendNames.OPENBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** LAPACKE resolves separately from CBLAS, either inside OpenBLAS or as its own library. */
    override val isAvailable: Boolean get() = HostBlasCalls.lapackAvailable

    override val nativeTrsmMinRhs: Int get() = NATIVE_TRSM_MIN_RHS

    override val choleskyMin: Int get() = JVM_CHOLESKY_MIN

    override val spdInvertMin: Int get() = JVM_INVERT_SPD_MIN

    override fun dgeqp3(m: Int, n: Int, a: DoubleArray, jpvt: IntArray, tau: DoubleArray): Int? {
        val dgeqp3 = HostBlasCalls.dgeqp3 ?: return null
        return dgeqp3.invokeWithArguments(
            COL_MAJOR,
            m,
            n,
            HostBlasCalls.seg(a),
            m,
            HostBlasCalls.seg(jpvt),
            HostBlasCalls.seg(tau),
        ) as Int
    }
}

/** Columns from which one blocked dtrsm beats a native call per column, measured on the JVM. */
private const val NATIVE_TRSM_MIN_RHS = 4

/** Where dpotrf takes over from the portable Cholesky. */
private const val JVM_CHOLESKY_MIN = 32

/** Where dpotri takes over from the portable SPD inverse. */
private const val JVM_INVERT_SPD_MIN = 16
