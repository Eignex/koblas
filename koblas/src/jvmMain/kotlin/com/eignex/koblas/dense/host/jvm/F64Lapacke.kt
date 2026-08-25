package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64DecompositionsAdapter
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.backend.hostBlasDispatchThresholds

/**
 * The host LAPACKE through `java.lang.foreign`. Every shared routine lives in [F64DecompositionsAdapter]; this
 * supplies the JVM's entry points and measured gates.
 */
public class F64Lapacke internal constructor(private val calls: HostBlasCalls, config: HostBlasConfig) :
    F64DecompositionsAdapter(
        JvmLapackeCalls(calls),
        JvmCblasCalls(calls),
        dispatch = hostBlasDispatchThresholds(config),
    ) {
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(HostBlasCalls(config), config)
    override val name: String get() = BackendNames.OPENBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** LAPACKE resolves separately from CBLAS, either inside OpenBLAS or as its own library. */
    override val isAvailable: Boolean get() = calls.lapackAvailable

    override fun dgeqp3(m: Int, n: Int, a: DoubleArray, jpvt: IntArray, tau: DoubleArray): Int? {
        val dgeqp3 = calls.dgeqp3 ?: return null
        return dgeqp3.invokeExact(
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
