package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.host.cblas.*
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
import com.eignex.koblas.sparse.host.cholmod.CholmodConfig
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig

/**
 * Backend discovery on the native targets that can reach a host library, run once on the first
 * [com.eignex.koblas.koblas] read. A resolved key symbol counts as installed, nothing is computed.
 *
 * Configuration comes from the environment, since Kotlin/Native has no system properties: the same variables
 * the JVM honours name a library path or pin a half to one backend.
 */
internal actual fun registerPlatformBackends() {
    val requested = requestedBackends()
    registerHostBlas(requested)
    registerSparse(requested)
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

/**
 * koblas's sparse bindings, each independent of the BLAS half and of each other: a host can have SuiteSparse
 * without BASICLU, or one of KLU and UMFPACK without the other. Priority picks the winner among those that
 * resolve, and only BASICLU offers basis updates whatever the priorities say.
 */
private fun registerSparse(requested: Map<BackendSlot, String?>) {
    val sparse = F64SparseBackends(
        kluConfig = KluConfig(libraryPath(ConfigurationKeys.KLU_PATH)),
        umfpackConfig = UmfpackConfig(libraryPath = libraryPath(ConfigurationKeys.UMFPACK_PATH)),
        basicluConfig = BasicluConfig(libraryPath(ConfigurationKeys.BASICLU_PATH)),
        cholmodConfig = CholmodConfig(libraryPath = libraryPath(ConfigurationKeys.CHOLMOD_PATH)),
    )
    for (backend in listOf(sparse.umfpack, sparse.klu, sparse.basiclu, sparse.cholmod)) {
        registerIfOffered(backend, requested)
    }
}
