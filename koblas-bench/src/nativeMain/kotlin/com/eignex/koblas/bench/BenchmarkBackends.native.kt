package com.eignex.koblas.bench

import com.eignex.koblas.dense.host.cblas.*
import com.eignex.koblas.installBackends
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.cholmod.CholmodSparseBlas
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
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

internal actual fun useSparseProduct(): Boolean {
    val cholmod = CholmodSparseBlas()
    if (!cholmod.isAvailable) return false
    installBackends(koblas.with(sparseBlas = cholmod))
    return true
}

internal actual fun useSparseLu(): Boolean {
    val umfpack = F64SparseBackends(umfpackConfig = UmfpackConfig()).umfpack
    if (!umfpack.isAvailable) return false
    installBackends(koblas.with(sparseDecompositions = umfpack))
    return true
}
