package com.eignex.koblas.hostblas

import com.eignex.koblas.BackendNames
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.HostBlasAdapter

/**
 * The host OpenBLAS through CBLAS, bound with `java.lang.foreign`. Every routine lives in
 * [HostBlasAdapter]; this supplies the JVM's entry points and the backend's identity.
 */
public class HostBlas internal constructor() : HostBlasAdapter(JvmCblasCalls) {
    override val name: String get() = BackendNames.OPENBLAS

    /** Above the reference (0) and the native dlopen backend (90). */
    override val priority: Int get() = HOST_BACKEND_PRIORITY
}
