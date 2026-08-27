package com.eignex.koblas.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
import com.eignex.koblas.sparse.host.basiclu.BasicluOptions
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu

/**
 * BASICLU from this module's bundled native resources, driven by koblas's own BASICLU binding. A deployment
 * pointing `koblas.basiclu.path` or `KOBLAS_BASICLU_PATH` at its own build gets that binding directly, and
 * discovery leaves this provider out of the running.
 *
 * One of [BasicluSparseLu] rather than a wrapper around one. BASICLU's own routines sit on that class rather
 * than on a seam, so a caller reaching it through [com.eignex.koblas.backendNamed] can only
 * use them if the bundled provider and the configured binding are the same type.
 */
public class BundledBasiclu private constructor(config: BasicluConfig) : BasicluSparseLu(config) {
    /** Creates bundled BASICLU with default options. */
    public constructor() : this(BasicluOptions())

    /** Creates bundled BASICLU with the same numerical and execution [options] accepted by the host binding. */
    public constructor(options: BasicluOptions) : this(
        BasicluConfig(basicluLibrary.extract().toString(), options),
    )

    /** Retains the original threshold-and-equilibration constructor. */
    public constructor(factorizeMin: Int?, equilibrate: Boolean = false) : this(
        BasicluOptions(factorizeMin = factorizeMin, equilibrate = equilibrate),
    )

    /** Retains calls that configured only equilibration by name. */
    public constructor(equilibrate: Boolean) : this(BasicluOptions(equilibrate = equilibrate))

    override val name: String get() = "basiclu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 3
}

private val basicluLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-basiclu",
    resourceRoot = "org/eignex/basiclu",
    anchor = BundledBasiclu::class.java,
    libraryDescription = "BASICLU",
    linuxSoname = "libkoblas_basiclu.so.1",
    macosSoname = "libkoblas_basiclu.1.dylib",
) { _, _ -> "koblas-basiclu has no bundled BASICLU for this host" }
