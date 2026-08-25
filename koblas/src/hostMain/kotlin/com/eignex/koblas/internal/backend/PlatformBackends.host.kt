package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.dense.host.cblas.*
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
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
    val denseRequested = requestedBackend(
        ConfigurationKeys.DENSE_BACKEND_PROPERTY,
        ConfigurationKeys.DENSE_BACKEND_ENVIRONMENT,
    )
    val sparseRequested = requestedBackend(
        ConfigurationKeys.SPARSE_BACKEND_PROPERTY,
        ConfigurationKeys.SPARSE_BACKEND_ENVIRONMENT,
    )
    registerHostBlas(denseRequested)
    registerSparse(sparseRequested)
}

/** koblas's CBLAS binding, when this host has OpenBLAS and the deployment did not pin another backend. */
private fun registerHostBlas(requested: String?) {
    val config = HostBlasConfig(
        libraryPath = libraryPath(ConfigurationKeys.CBLAS_PATH),
        lapackeLibraryPath = libraryPath(ConfigurationKeys.LAPACKE_PATH),
    )
    val loader = OpenBlasLoader(config)
    val cblas = loader.cblas ?: return
    val blas = F64Cblas(cblas, loader, config)
    if (!offered(blas, requested)) return
    BackendRegistry.registerAutomatic(blas)
    // The level-1 primitives sit below the F64Blas seam, so they register as their own half.
    BackendRegistry.registerAutomatic(F64CblasKernels(loader, config))
    // Without LAPACKE the factorizations stay portable while everything above keeps the host BLAS.
    val lapacke = loader.lapacke ?: return
    BackendRegistry.registerAutomatic(F64Lapacke(lapacke, cblas, loader, config))
}

/**
 * koblas's sparse bindings, each independent of the BLAS half and of each other: a host can have SuiteSparse
 * without BASICLU, or one of KLU and UMFPACK without the other. Priority picks the winner among those that
 * resolve, and only BASICLU offers basis updates whatever the priorities say.
 */
private fun registerSparse(requested: String?) {
    val sparse = F64SparseBackends(
        umfpackConfig = UmfpackConfig(libraryPath = libraryPath(ConfigurationKeys.UMFPACK_PATH)),
        kluConfig = KluConfig(libraryPath(ConfigurationKeys.KLU_PATH)),
        basicluConfig = BasicluConfig(libraryPath(ConfigurationKeys.BASICLU_PATH)),
    )
    for (backend in listOf(sparse.umfpack, sparse.klu, sparse.basiclu)) {
        if (backend.isAvailable && offered(backend, requested)) BackendRegistry.registerAutomatic(backend)
    }
}

/** Whether [backend] is what the deployment asked for, or asked for nothing in particular. */
private fun offered(backend: Backend, requested: String?): Boolean =
    requested == null || matchesRequested(backend.name, requested)
