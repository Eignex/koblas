package com.eignex.koblas.bench

import com.eignex.koblas.dense.host.cblas.*
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

internal actual fun useHost(): Boolean = installHost(HostBlasConfig())

// Constructed explicitly rather than left to discovery, because the linker may drop the unreferenced
// eager-init property and leave a silent reference-versus-reference run.
private fun installHost(config: HostBlasConfig): Boolean {
    if (!F64CblasBackend.isAvailable()) return false
    val backend = F64CblasBackend(config)
    installBackends(koblas.with(kernels = F64CblasKernels(config), blas = backend, decompositions = backend))
    return true
}
