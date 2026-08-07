package com.eignex.koblas.umfpack

import com.eignex.koblas.registerBackend

/**
 * Registers the UMFPACK sparse half at program start when the host has SuiteSparse, mirroring the CBLAS
 * backend's `cblasAutoInstall` and the JVM's ServiceLoader discovery.
 *
 * Separate from the CBLAS registration because the two libraries are independent: a host may have OpenBLAS
 * without SuiteSparse or the reverse, and each half should accelerate on its own. Nothing is registered when
 * the library is absent, and `koblas` stays on the portable `SparseLu`.
 *
 * No probe computation runs here, unlike the CBLAS auto-install. Resolution has already checked that the
 * library opens and carries `umfpack_di_symbolic`, and a probe factorization would be real work inside
 * program startup for a backend whose `factor` falls back on any failure anyway.
 *
 * If the linker drops this unreferenced property, `installBackends(koblas.with(sparseLapack = ...))` remains
 * the explicit path.
 */
@Suppress("unused")
@OptIn(ExperimentalStdlibApi::class)
@EagerInitialization
val umfpackAutoInstall: Unit = autoInstall()

private fun autoInstall() {
    val functions = UmfpackLoader.functions ?: return
    registerBackend(UmfpackSparseLapack(functions))
}
