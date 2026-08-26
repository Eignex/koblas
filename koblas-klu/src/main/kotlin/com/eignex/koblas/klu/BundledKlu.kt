package com.eignex.koblas.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu

/**
 * KLU backend bundled in Maven-native resources.
 *
 * One of [KluSparseLu] rather than a wrapper around one, so the analysis reuse that sits on that class stays
 * reachable through [com.eignex.koblas.sparseDecompositionsNamed] whichever of the two answered to the name.
 */
public class BundledKlu(factorizeMin: Int? = null, equilibrate: Boolean = false) :
    KluSparseLu(
        KluConfig(kluLibrary.extract().toString(), factorizeMin, equilibrate),
    ) {
    override val name: String get() = "klu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
}

private val kluLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-klu",
    resourceRoot = "org/eignex/klu",
    anchor = BundledKlu::class.java,
    libraryDescription = "KLU",
    linuxSoname = "libklu.so.2",
    macosSoname = "libklu.2.dylib",
) { _, _ -> "koblas-klu has no bundled KLU for this host" }
