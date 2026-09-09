package com.eignex.koblas.hfactor

import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorOptions
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu

/**
 * HiGHS's HFactor from this module's bundled native resources. Use [HfactorSparseLu] with an explicit
 * [HfactorConfig.libraryPath] to load another build of the same implementation.
 */
public class BundledHfactor private constructor(config: HfactorConfig) : HfactorSparseLu(config) {
    /** Creates bundled HFactor with default options. */
    constructor() : this(HfactorOptions())

    /** Creates bundled HFactor with the same numerical and execution [options] accepted by the host binding. */
    constructor(options: HfactorOptions) : this(
        HfactorConfig(hfactorLibrary.extract().toString(), options),
    )
}

private val hfactorLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-hfactor",
    resourceRoot = "org/eignex/hfactor",
    anchor = BundledHfactor::class.java,
    libraryDescription = "HFactor",
    linuxSoname = "libkoblas_hfactor.so.1",
    macosSoname = "libkoblas_hfactor.1.dylib",
) { _, _ -> "koblas-hfactor has no bundled HFactor for this host" }
