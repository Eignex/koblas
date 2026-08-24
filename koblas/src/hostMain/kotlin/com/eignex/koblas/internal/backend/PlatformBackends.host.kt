package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.host.cblas.F64CblasBlas
import com.eignex.koblas.dense.host.cblas.F64CblasLapack
import com.eignex.koblas.dense.host.cblas.F64CblasVectorKernels
import com.eignex.koblas.dense.host.cblas.OpenBlasConfig
import com.eignex.koblas.dense.host.cblas.OpenBlasLoader
import com.eignex.koblas.sparse.host.F64HostSparseBackends

/**
 * Backend discovery on the native targets that can reach a host library, run once on the first
 * [com.eignex.koblas.koblas] read. A resolved key symbol counts as installed, nothing is computed.
 */
internal actual fun registerPlatformBackends() {
    registerHostBlas()
    registerUmfpack()
}

/** koblas's CBLAS binding, when this host has OpenBLAS. */
private fun registerHostBlas() {
    val config = OpenBlasConfig()
    val loader = OpenBlasLoader(config)
    val cblas = loader.cblas ?: return
    BackendRegistry.registerAutomatic(F64CblasBlas(cblas, loader, config))
    // The level-1 primitives sit below the F64Blas seam, so they register as their own half.
    BackendRegistry.registerAutomatic(F64CblasVectorKernels(loader, config))
    // Without LAPACKE the factorizations stay portable while everything above keeps the host BLAS.
    val lapacke = loader.lapacke ?: return
    BackendRegistry.registerAutomatic(F64CblasLapack(lapacke, cblas, loader, config))
}

/** koblas's UMFPACK binding, when this host has SuiteSparse. Independent of the BLAS half by design. */
private fun registerUmfpack() {
    val umfpack = F64HostSparseBackends().umfpack
    if (umfpack.isAvailable) BackendRegistry.registerAutomatic(umfpack)
}
