package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64BlasAdapter
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.backend.hostBlasDispatchThresholds

/**
 * The host OpenBLAS through CBLAS, bound with cinterop. Every routine lives in [F64BlasAdapter]; this
 * supplies the native entry points and the backend's identity. Constructible whenever the host has
 * OpenBLAS, independently of LAPACKE.
 */
internal class F64Cblas(
    f: CblasFunctions,
    private val loader: OpenBlasLoader = OpenBlasLoader(),
    config: HostBlasConfig = HostBlasConfig(),
) : F64BlasAdapter(
    NativeCblasCalls(f),
    dispatch = hostBlasDispatchThresholds(config),
    metadata = BackendMetadata(
        integerAbi = "LP64",
        threading = loader.effectiveThreadCount?.let { "$it threads" },
        options = config.options.metadataOptions(loader.effectiveThreadCount),
    ),
) {
    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** The BLAS half needs only CBLAS, which a host can provide without LAPACKE. */
    override val isAvailable: Boolean get() = loader.cblas != null
}
