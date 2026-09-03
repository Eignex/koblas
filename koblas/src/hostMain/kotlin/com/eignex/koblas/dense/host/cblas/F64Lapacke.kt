@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64DecompositionsAdapter
import com.eignex.koblas.internal.backend.BackendNames
import kotlinx.cinterop.*

/**
 * The host LAPACKE through cinterop. Every shared routine lives in [F64DecompositionsAdapter]; this supplies the
 * native entry points and the backend's identity. The gates come from the configuration, as they do for the
 * JVM binding.
 */
internal class F64Lapacke(
    f: LapackeFunctions,
    blas: CblasFunctions,
    private val loader: OpenBlasLoader = OpenBlasLoader(),
    config: HostBlasConfig = HostBlasConfig(),
) : F64DecompositionsAdapter(
    NativeLapackeCalls(f),
    NativeCblasCalls(blas),
    metadata = BackendMetadata(
        integerAbi = "LP64",
        threading = loader.effectiveThreadCount?.let { "$it threads" },
        options = config.options.metadataOptions(loader.effectiveThreadCount),
    ),
) {
    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** LAPACKE calls into CBLAS, so this half needs both. */
    override val isAvailable: Boolean get() = loader.lapacke != null && loader.cblas != null
}
