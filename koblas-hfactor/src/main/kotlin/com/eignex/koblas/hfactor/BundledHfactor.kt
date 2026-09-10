package com.eignex.koblas.hfactor

import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu

/**
 * HiGHS's HFactor from this module's bundled native resources. Use [HfactorSparseLu] with an explicit
 * [HfactorConfig.libraryPath] to load another build of the same implementation. [config] may select the
 * numerical policy, but its library path must be null because this class always loads the bundled library.
 */
public class BundledHfactor(config: HfactorConfig = HfactorConfig()) : HfactorSparseLu(config.withBundledLibrary())

private fun HfactorConfig.withBundledLibrary(): HfactorConfig {
    require(libraryPath == null) {
        "BundledHfactor does not accept libraryPath; use HfactorSparseLu for an explicit library"
    }
    return copy(libraryPath = hfactorLibrary.extract().toString())
}

private val hfactorLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-hfactor",
    resourceRoot = "org/eignex/hfactor",
    anchor = BundledHfactor::class.java,
    libraryDescription = "HFactor",
    linuxSoname = "libkoblas_hfactor.so.1",
    macosSoname = "libkoblas_hfactor.1.dylib",
) { _, _ -> "koblas-hfactor has no bundled HFactor for this host" }
