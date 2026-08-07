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
 * [com.eignex.koblas.koblas] read.
 *
 * The same shape and the same timing as the JVM's discovery: each library is resolved by soname, a key symbol
 * decides whether it counts as installed, and nothing is computed to confirm it. Registration is per half, so
 * a machine with OpenBLAS and no SuiteSparse gets native dense routines and the portable sparse factorization,
 * and the reverse combination works too.
 *
 * This exists as its own source set because it is the only place that can see both bindings: `hostBlasMain`
 * and `hostSparseMain` are siblings, and Linux and macOS are the targets that depend on both.
 *
 * Replaces an `@EagerInitialization` property per binding, which ran before `main`. That annotation is
 * deprecated, and it carried a hazard of its own — the property is unreferenced, so a linker free to drop it
 * would have silently produced a binary with no host backend at all. Discovery on first use has neither
 * problem, and it makes the two platforms answer "when does a backend register" the same way.
 */
internal actual fun registerPlatformBackends() {
    registerHostBlas()
    registerUmfpack()
}

/** koblas's CBLAS binding, when this host has OpenBLAS. */
private fun registerHostBlas() {
    val cblas = OpenBlasLoader.cblas ?: return
    registerBackend(CblasBlas(cblas))
    // The level-1 primitives sit below the Blas seam and are scalar on this platform, so they register as
    // their own half; koblas routes only runs long enough to cover the call.
    registerBackend(CblasVectorKernels())
    // LAPACKE is a separate package on Debian and Ubuntu. Without it the factorizations stay portable while
    // everything above keeps the host BLAS, rather than the whole backend dropping out.
    val lapacke = OpenBlasLoader.lapacke ?: return
    registerBackend(CblasLapack(lapacke, cblas))
}

/** koblas's UMFPACK binding, when this host has SuiteSparse. Independent of the BLAS half by design. */
private fun registerUmfpack() {
    val functions = UmfpackLoader.functions ?: return
    registerBackend(UmfpackSparseLapack(functions))
}
