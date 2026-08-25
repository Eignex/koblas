package com.eignex.koblas.bench

import com.eignex.koblas.dense.host.cblas.*
import com.eignex.koblas.installBackends
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.koblas

internal actual fun useShippedHost(): Boolean = installHost(HostBlasConfig())

// Level 2, level 3 and the factorizations already dispatch to the host from size zero here, so only the
// level-1 gate needs overriding; the kernels are installed unrouted because the benchmark sweeps the
// lengths itself.
internal actual fun useUngatedHost(): Boolean =
    installHost(HostBlasConfig(level1Min = 0, level2Min = 0, level3Min = 0, factorizeMin = 0))

// Constructed explicitly rather than left to discovery, because the linker may drop the unreferenced
// eager-init property and leave a silent reference-versus-reference run.
private fun installHost(config: HostBlasConfig): Boolean {
    if (!F64CblasBackend.isAvailable()) return false
    val backend = F64CblasBackend(config)
    installBackends(koblas.with(kernels = F64CblasKernels(config), blas = backend, decompositions = backend))
    return true
}

internal actual fun useUngatedSparseLu(): Boolean {
    val umfpack = F64SparseBackends(UmfpackConfig(factorizeMin = 0)).umfpack
    if (!umfpack.isAvailable) return false
    installBackends(koblas.with(sparseLu = umfpack))
    return true
}
