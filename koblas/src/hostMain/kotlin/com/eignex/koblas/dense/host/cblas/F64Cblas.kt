package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64BlasAdapter
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The host OpenBLAS through CBLAS, bound with cinterop. Every routine lives in [F64BlasAdapter]; this
 * supplies the native entry points and the backend's identity.
 */
internal class F64Cblas(
    f: CblasFunctions,
    private val loader: OpenBlasLoader = OpenBlasLoader(),
    config: HostBlasConfig = HostBlasConfig(),
) : F64BlasAdapter(
    NativeCblasCalls(f),
    metadata = BackendMetadata(
        integerAbi = "LP64",
        threading = loader.effectiveThreadCount?.let { "$it threads" },
        options = config.options.metadataOptions(loader.effectiveThreadCount),
    ),
) {
    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** Whether the host provides the complete CBLAS subset. */
    override val isAvailable: Boolean get() = loader.cblas != null
}
