package com.eignex.koblas.suitesparse

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.openblas.BundledOpenBlas
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/**
 * SuiteSparse UMFPACK backend bundled in Maven-native resources.
 *
 * One of [UmfpackSparseLu] rather than a wrapper around one, so the bundled providers all answer to
 * [com.eignex.koblas.sparseDecompositionsNamed] as the type their binding is.
 */
public class BundledUmfpack(factorizeMin: Int? = null, equilibrate: Boolean = false) :
    UmfpackSparseLu(bundledConfig(factorizeMin, equilibrate)) {
    override val name: String get() = "umfpack-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
}

/** UMFPACK is linked against OpenBLAS, so the dependency is checked before the library is extracted. */
private fun bundledConfig(factorizeMin: Int?, equilibrate: Boolean): UmfpackConfig {
    check(BundledOpenBlas().isAvailable) { "the bundled OpenBLAS dependency could not be loaded" }
    return UmfpackConfig(
        libraryPath = umfpackLibrary.extract().toString(),
        factorizeMin = factorizeMin,
        equilibrate = equilibrate,
    )
}

private val umfpackLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-suitesparse",
    resourceRoot = "org/eignex/suitesparse",
    anchor = BundledUmfpack::class.java,
    libraryDescription = "UMFPACK",
    linuxSoname = "libumfpack.so.6",
    macosSoname = "libumfpack.dylib",
) { _, _ -> "koblas-suitesparse has no bundled UMFPACK for this host" }
