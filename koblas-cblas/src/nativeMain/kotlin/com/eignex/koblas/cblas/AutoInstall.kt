package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.registerLinearAlgebra

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
    if (!CblasLinearAlgebra.isAvailable()) return
    val backend = CblasLinearAlgebra()
    // The same guard the JVM discovery applies: a tiny computation must come back correct.
    if (probe(backend)) registerLinearAlgebra(backend)
}

@Suppress("TooGenericExceptionCaught", "SwallowedException") // a broken installation must not crash startup
private fun probe(backend: LinearAlgebra): Boolean = try {
    val y = backend.gemv(DenseMatrix.wrap(1, 1, doubleArrayOf(2.0)), doubleArrayOf(3.0))
    y.size == 1 && y[0] == 6.0
} catch (e: Throwable) {
    false
}
