package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64HostLapackAdapter
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.dense.host.cblas.OpenBlasConfig
import com.eignex.koblas.dense.host.jvm.HostBlasCalls
import com.eignex.koblas.dense.host.jvm.JvmCblasCalls
import com.eignex.koblas.dense.host.jvm.JvmLapackeCalls
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.backend.openBlasDispatchThresholds

/**
 * The host LAPACKE through `java.lang.foreign`. Every shared routine lives in [F64HostLapackAdapter]; this
 * supplies the JVM's entry points and measured gates.
 */
public class F64OpenBlasLapack internal constructor(private val calls: HostBlasCalls, config: OpenBlasConfig) :
    F64HostLapackAdapter(
        JvmLapackeCalls(calls),
        JvmCblasCalls(calls),
        dispatch = openBlasDispatchThresholds(config),
    ) {
    public constructor(config: OpenBlasConfig = OpenBlasConfig()) : this(HostBlasCalls(config), config)
    override val name: String get() = BackendNames.OPENBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** LAPACKE resolves separately from CBLAS, either inside OpenBLAS or as its own library. */
    override val isAvailable: Boolean get() = calls.lapackAvailable

    override fun dgeqp3(m: Int, n: Int, a: DoubleArray, jpvt: IntArray, tau: DoubleArray): Int? {
        val dgeqp3 = calls.dgeqp3 ?: return null
        return dgeqp3.invokeWithArguments(
            COL_MAJOR,
            m,
            n,
            java.lang.foreign.MemorySegment.ofArray(a),
            m,
            java.lang.foreign.MemorySegment.ofArray(jpvt),
            java.lang.foreign.MemorySegment.ofArray(tau),
        ) as Int
    }
}
