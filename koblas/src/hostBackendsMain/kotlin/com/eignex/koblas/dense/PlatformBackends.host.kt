package com.eignex.koblas.dense

import com.eignex.koblas.cblas.F64CblasVectorKernels
import com.eignex.koblas.dense.host.cblas.F64CblasBlas
import com.eignex.koblas.dense.host.cblas.F64CblasLapack
import com.eignex.koblas.dense.host.cblas.OpenBlasLoader
import com.eignex.koblas.registerBackend
import com.eignex.koblas.sparse.host.umfpack.UmfpackLoader
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLapack

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
    val cblas = OpenBlasLoader.cblas ?: return
    registerBackend(F64CblasBlas(cblas))
    // The level-1 primitives sit below the F64Blas seam, so they register as their own half.
    registerBackend(F64CblasVectorKernels())
    // Without LAPACKE the factorizations stay portable while everything above keeps the host BLAS.
    val lapacke = OpenBlasLoader.lapacke ?: return
    registerBackend(F64CblasLapack(lapacke, cblas))
}

/** koblas's UMFPACK binding, when this host has SuiteSparse. Independent of the BLAS half by design. */
private fun registerUmfpack() {
    val functions = UmfpackLoader.functions ?: return
    registerBackend(UmfpackSparseLapack(functions))
}
