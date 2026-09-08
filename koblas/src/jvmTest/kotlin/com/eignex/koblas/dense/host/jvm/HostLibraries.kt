package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.cblas.HostBlasConfig

/**
 * Whether this host offers OpenBLAS, resolved once for the tests that skip themselves without it. A
 * production backend owns its own calls, built from its own configuration, so this lives here rather than
 * as a static on the binding.
 */
internal object HostLibraries {
    private val calls by lazy { HostBlasCalls(HostBlasConfig()) }

    /** Whether the host's CBLAS resolved. */
    val cblas: Boolean get() = calls.available
}
