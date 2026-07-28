package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.installLinearAlgebra

/**
 * Installs [CblasLinearAlgebra] as the active backend at program start when the host provides
 * OpenBLAS — depending on the koblas-cblas artifact activates it, mirroring the JVM's classpath
 * discovery. On hosts without the libraries nothing is installed and [com.eignex.koblas.koblas]
 * stays on the portable reference. Call [installLinearAlgebra] to override either way; if the
 * linker drops this unreferenced property, `installLinearAlgebra(CblasLinearAlgebra())` remains
 * the explicit activation path.
 */
@Suppress("unused")
@OptIn(ExperimentalStdlibApi::class)
@EagerInitialization
val cblasAutoInstall: Unit = autoInstall()

private fun autoInstall() {
    if (!CblasLinearAlgebra.isAvailable()) return
    val backend = CblasLinearAlgebra()
    // The same guard the JVM discovery applies: a tiny computation must come back correct.
    if (probe(backend)) installLinearAlgebra(backend)
}

@Suppress("TooGenericExceptionCaught", "SwallowedException") // a broken installation must not crash startup
private fun probe(backend: LinearAlgebra): Boolean = try {
    val y = backend.gemv(DenseMatrix.wrap(1, 1, doubleArrayOf(2.0)), doubleArrayOf(3.0))
    y.size == 1 && y[0] == 6.0
} catch (e: Throwable) {
    false
}
