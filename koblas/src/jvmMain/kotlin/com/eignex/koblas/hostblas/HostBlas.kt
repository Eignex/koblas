package com.eignex.koblas.hostblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64HostBlasAdapter
import com.eignex.koblas.dense.host.cblas.OpenBlasConfig
import com.eignex.koblas.dense.host.jvm.HostBlasCalls
import com.eignex.koblas.dense.host.jvm.JvmCblasCalls
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The host OpenBLAS through CBLAS, bound with `java.lang.foreign`. Every routine lives in
 * [F64HostBlasAdapter]; this supplies the JVM's entry points and the backend's identity.
 */
public class HostBlas internal constructor(private val calls: HostBlasCalls) :
    F64HostBlasAdapter(
        JvmCblasCalls(calls),
    ) {
    public constructor(config: OpenBlasConfig = OpenBlasConfig()) : this(HostBlasCalls(config))
    override val name: String get() = BackendNames.OPENBLAS

    /** Above the reference (0) and the native dlopen backend (90). */
    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** The BLAS half needs only CBLAS, which a host can provide without LAPACKE. */
    override val isAvailable: Boolean get() = calls.available
}
