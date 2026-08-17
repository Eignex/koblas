package com.eignex.koblas.cblas

import com.eignex.koblas.BackendNames
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.HostLapackAdapter

/**
 * The host LAPACKE through cinterop. Every routine lives in [HostLapackAdapter]; this supplies the native
 * entry points and the backend's identity. The gates keep their defaults, so native dispatches at any size.
 */
internal class CblasLapack(f: LapackeFunctions, blas: CblasFunctions) :
    HostLapackAdapter(NativeLapackeCalls(f), NativeCblasCalls(blas)) {
    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY
}
