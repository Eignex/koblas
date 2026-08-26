@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.host.F64DecompositionsAdapter
import com.eignex.koblas.dense.host.cblas.Cblas.COL_MAJOR
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.backend.hostBlasDispatchThresholds
import kotlinx.cinterop.*

/**
 * The host LAPACKE through cinterop. Every shared routine lives in [F64DecompositionsAdapter]; this supplies the
 * native entry points and the backend's identity. The gates come from the configuration, as they do for the
 * JVM binding.
 */
internal class F64Lapacke(
    private val f: LapackeFunctions,
    blas: CblasFunctions,
    private val loader: OpenBlasLoader = OpenBlasLoader(),
    config: HostBlasConfig = HostBlasConfig(),
) : F64DecompositionsAdapter(
    NativeLapackeCalls(f),
    NativeCblasCalls(blas),
    dispatch = hostBlasDispatchThresholds(config),
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

    override fun dgeqp3(m: Int, n: Int, a: DoubleArray, jpvt: IntArray, tau: DoubleArray): Int? {
        val dgeqp3 = f.dgeqp3 ?: return null
        return a.usePinned { ap ->
            jpvt.usePinned { jp ->
                tau.usePinned { tp -> dgeqp3(COL_MAJOR, m, n, ap.addressOf(0), m, jp.addressOf(0), tp.addressOf(0)) }
            }
        }
    }
}
