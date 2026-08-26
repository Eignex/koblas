package com.eignex.koblas.hfactor

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu

/**
 * HiGHS's HFactor from this module's bundled native resources, driven by koblas's own HFactor binding. A
 * deployment pointing `koblas.hfactor.path` or `KOBLAS_HFACTOR_PATH` at its own build gets that binding
 * directly, and discovery leaves this provider out of the running.
 *
 * One of [HfactorSparseLu] rather than a wrapper around one, which is how the bundled providers reach the
 * routines their bindings carry outside a seam.
 */
public class BundledHfactor(factorizeMin: Int? = null, equilibrate: Boolean = false) :
    HfactorSparseLu(
        HfactorConfig(hfactorLibrary.extract().toString(), factorizeMin, equilibrate),
    ) {
    override val name: String get() = "hfactor-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY - 1
}

private val hfactorLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-hfactor",
    resourceRoot = "org/eignex/hfactor",
    anchor = BundledHfactor::class.java,
    libraryDescription = "HFactor",
    linuxSoname = "libkoblas_hfactor.so.1",
    macosSoname = "libkoblas_hfactor.1.dylib",
) { _, _ -> "koblas-hfactor has no bundled HFactor for this host" }
