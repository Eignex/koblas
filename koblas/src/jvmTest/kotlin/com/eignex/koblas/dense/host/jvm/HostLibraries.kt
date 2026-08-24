package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.cblas.OpenBlasConfig

/**
 * What this host's OpenBLAS offers, resolved once for the tests that skip themselves without it. A
 * production backend owns its own calls, built from its own configuration, so this lives here rather than
 * as a static on the binding.
 */
internal object HostLibraries {
    private val calls by lazy { HostBlasCalls(OpenBlasConfig()) }

    /** Whether the host's CBLAS resolved. */
    val cblas: Boolean get() = calls.available

    /** Whether the host's LAPACKE resolved as well. */
    val lapacke: Boolean get() = calls.lapackAvailable
}
