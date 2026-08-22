package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.F64HostBlasAdapter
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The host OpenBLAS through CBLAS, bound with cinterop. Every routine lives in [F64HostBlasAdapter]; this
 * supplies the native entry points and the backend's identity. Constructible whenever the host has
 * OpenBLAS, independently of LAPACKE.
 */
internal class F64CblasBlas(f: CblasFunctions) : F64HostBlasAdapter(NativeCblasCalls(f)) {
    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** The BLAS half needs only CBLAS, which a host can provide without LAPACKE. */
    override val isAvailable: Boolean get() = OpenBlasLoader.cblas != null
}
