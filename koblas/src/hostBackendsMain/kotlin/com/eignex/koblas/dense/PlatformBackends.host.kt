package com.eignex.koblas.dense

import com.eignex.koblas.cblas.CblasBlas
import com.eignex.koblas.cblas.CblasLapack
import com.eignex.koblas.cblas.CblasVectorKernels
import com.eignex.koblas.cblas.OpenBlasLoader
import com.eignex.koblas.registerBackend
import com.eignex.koblas.umfpack.UmfpackLoader
import com.eignex.koblas.umfpack.UmfpackSparseLapack

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
    registerBackend(CblasBlas(cblas))
    // The level-1 primitives sit below the Blas seam, so they register as their own half.
    registerBackend(CblasVectorKernels())
    // Without LAPACKE the factorizations stay portable while everything above keeps the host BLAS.
    val lapacke = OpenBlasLoader.lapacke ?: return
    registerBackend(CblasLapack(lapacke, cblas))
}

/** koblas's UMFPACK binding, when this host has SuiteSparse. Independent of the BLAS half by design. */
private fun registerUmfpack() {
    val functions = UmfpackLoader.functions ?: return
    registerBackend(UmfpackSparseLapack(functions))
}
