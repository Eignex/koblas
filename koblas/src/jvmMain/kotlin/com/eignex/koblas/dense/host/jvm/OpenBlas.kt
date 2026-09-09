package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.BlasAdapter
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.cblas.metadataOptions
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The host OpenBLAS through CBLAS, bound with `java.lang.foreign`. Every routine lives in
 * [BlasAdapter]; this supplies the JVM's entry points and the backend's identity.
 */
public class OpenBlas internal constructor(private val calls: HostBlasCalls) :
    BlasAdapter(
        JvmCblasCalls(calls),
        metadata = BackendMetadata(
            integerAbi = "LP64",
            threading = calls.effectiveThreadCount?.let { "$it threads" },
            options = calls.config.options.metadataOptions(calls.effectiveThreadCount),
        ),
    ) {
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(HostBlasCalls(config))
    override val name: String get() = BackendNames.OPENBLAS

    /** Above the reference (0) and the native dlopen backend (90). */
    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** Whether the host provides the complete CBLAS subset. */
    override val isAvailable: Boolean get() = calls.available
}
