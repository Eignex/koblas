package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.host.cblas.*

/**
 * Backend discovery on the native targets that can reach a host library, run once on the first
 * [com.eignex.koblas.koblas] read. A resolved key symbol counts as installed, nothing is computed.
 *
 * Configuration comes from the environment, since Kotlin/Native has no system properties: the same variables
 * the JVM honours name a library path or pin a half to one backend. The sparse halves stay portable here,
 * since the one sparse binding koblas carries has no native target.
 */
internal actual fun registerPlatformBackends() {
    registerHostBlas(requestedBackends())
}

/** koblas's CBLAS binding, when this host has OpenBLAS and the deployment did not pin another backend. */
private fun registerHostBlas(requested: Map<BackendSlot, String?>) {
    val config = HostBlasConfig(
        libraryPath = libraryPath(ConfigurationKeys.CBLAS_PATH),
        lapackeLibraryPath = libraryPath(ConfigurationKeys.LAPACKE_PATH),
    )
    val loader = OpenBlasLoader(config)
    val cblas = loader.cblas ?: return
    val blas = F64Cblas(cblas, loader, config)
    registerIfOffered(blas, requested)
    // The level-1 primitives sit below the F64Blas seam, so they register as their own half.
    registerIfOffered(F64CblasKernels(loader, config), requested)
    // Without LAPACKE the factorizations stay portable while everything above keeps the host BLAS.
    val lapacke = loader.lapacke ?: return
    registerIfOffered(F64Lapacke(lapacke, cblas, loader, config), requested)
}
