package com.eignex.koblas.cblas

import com.eignex.koblas.Blas
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.registerBlas
import com.eignex.koblas.registerLapack
import com.eignex.koblas.registerVectorKernels

/**
 * Registers [CblasLinearAlgebra] for automatic selection at program start when the host provides
 * OpenBLAS — depending on the koblas-cblas artifact activates it, mirroring the JVM's classpath
 * discovery, while priority ranking keeps any stronger registered backend in front. On hosts
 * without the libraries nothing is registered and [com.eignex.koblas.koblas] stays on the portable
 * reference. `installLinearAlgebra` overrides either way; if the linker drops this unreferenced
 * property, `installLinearAlgebra(CblasLinearAlgebra())` remains the explicit activation path.
 */
@Suppress("unused")
@OptIn(ExperimentalStdlibApi::class)
@EagerInitialization
val cblasAutoInstall: Unit = autoInstall()

private fun autoInstall() {
    val cblas = OpenBlasLoader.cblas ?: return
    val blas = CblasBlas(cblas)
    // The same guard the JVM discovery applies: a tiny computation must come back correct.
    if (!probe(blas)) return
    registerBlas(blas)
    // The level-1 primitives sit below the Blas seam and are scalar on this platform, so they register as
    // their own half; koblas routes only runs long enough to cover the call.
    registerVectorKernels(CblasVectorKernels())
    // LAPACKE is a separate package on Debian and Ubuntu. Without it the factorizations stay portable
    // while everything above keeps the host BLAS, rather than the whole backend dropping out.
    val lapacke = OpenBlasLoader.lapacke ?: return
    registerLapack(CblasLapack(lapacke, cblas))
}

@Suppress("TooGenericExceptionCaught", "SwallowedException") // a broken installation must not crash startup
private fun probe(backend: Blas): Boolean = try {
    val y = backend.gemv(DenseMatrix.wrap(1, 1, doubleArrayOf(2.0)), doubleArrayOf(3.0))
    y.size == 1 && y[0] == 6.0
} catch (e: Throwable) {
    false
}
