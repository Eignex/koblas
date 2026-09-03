package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64DecompositionsAdapter
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.cblas.metadataOptions
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The host LAPACKE through `java.lang.foreign`. Every shared routine lives in [F64DecompositionsAdapter]; this
 * supplies the JVM's entry points and measured gates.
 */
public class F64Lapacke internal constructor(private val calls: HostBlasCalls, config: HostBlasConfig) :
    F64DecompositionsAdapter(
        JvmLapackeCalls(calls),
        JvmCblasCalls(calls),
        metadata = BackendMetadata(
            integerAbi = "LP64",
            threading = calls.effectiveThreadCount?.let { "$it threads" },
            options = config.options.metadataOptions(calls.effectiveThreadCount),
        ),
    ) {
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(HostBlasCalls(config), config)
    override val name: String get() = BackendNames.OPENBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** LAPACKE resolves separately from CBLAS, either inside OpenBLAS or as its own library. */
    override val isAvailable: Boolean get() = calls.lapackAvailable
}
