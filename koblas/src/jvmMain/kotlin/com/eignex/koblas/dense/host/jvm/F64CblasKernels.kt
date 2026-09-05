package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.F64KernelsAdapter
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The level-1 CBLAS routines from the OpenBLAS instance shared with [F64Cblas]. Every routine lives in
 * [F64KernelsAdapter]; this supplies the JVM's entry points and the backend's identity.
 */
public class F64CblasKernels internal constructor(
    private val calls: HostBlasCalls,
    /** Policy for this level-1 backend instance. */
    public val config: HostBlasConfig,
) : F64KernelsAdapter(JvmCblasKernelCalls(calls)) {
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(HostBlasCalls(config), config)

    override val name: String get() = BackendNames.OPENBLAS

    override val isAvailable: Boolean get() = calls.available
}
