package com.eignex.koblas.bench

import com.eignex.koblas.dense.host.cblas.*
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

internal actual fun useHost(): Boolean = installHost(HostBlasConfig())

// Read from the binding rather than written out, so the name an arm is checked against is the one the
// backend reports.
internal actual val hostBackendName: String get() = F64CblasBackend(HostBlasConfig()).name

// Constructed explicitly rather than left to discovery, because the linker may drop the unreferenced
// eager-init property and leave a silent reference-versus-reference run.
private fun installHost(config: HostBlasConfig): Boolean {
    if (!F64CblasBackend.isAvailable()) return false
    val backend = F64CblasBackend(config)
    installBackends(koblas.with(kernels = F64CblasKernels(config), blas = backend, decompositions = backend))
    return true
}
