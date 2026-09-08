package com.eignex.koblas.bench

import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.F64Backends
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

// The CBLAS and LAPACKE halves are separate objects here, so there is no single F64LinearAlgebra to hand
// back and every half is installed explicitly.
internal actual fun useHost(): Boolean = installHost(HostBlasConfig())

// Read from the binding rather than written out, so the name an arm is checked against is the one the
// backend reports.
internal actual val hostBackendName: String get() = F64Backends(HostBlasConfig()).blas.name

private fun installHost(config: HostBlasConfig): Boolean {
    val backends = F64Backends(config)
    if (!backends.blas.isAvailable) return false
    installBackends(koblas.with(blas = backends.blas, decompositions = backends.decompositions))
    return true
}
