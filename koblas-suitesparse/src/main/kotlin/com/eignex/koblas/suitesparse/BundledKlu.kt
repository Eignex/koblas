package com.eignex.koblas.suitesparse

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluOptions
import com.eignex.koblas.sparse.host.klu.KluSparseLu

/**
 * KLU backend bundled in Maven-native resources.
 *
 * One of [KluSparseLu] rather than a wrapper around one, so the analysis reuse that sits on that class stays
 * reachable through [com.eignex.koblas.backendNamed] whichever of the two answered to the name.
 */
class BundledKlu private constructor(config: KluConfig) : KluSparseLu(config) {
    /** Creates bundled KLU with default options. */
    constructor() : this(KluOptions())

    /** Creates bundled KLU with the same numerical and execution [options] accepted by the host binding. */
    constructor(options: KluOptions) : this(KluConfig(kluLibrary.extract().toString(), options))

    override val name: String get() = "klu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
}

private val kluLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-suitesparse",
    resourceRoot = "org/eignex/suitesparse",
    anchor = BundledKlu::class.java,
    libraryDescription = "KLU",
    linuxSoname = "libklu.so.2",
    macosSoname = "libklu.2.dylib",
) { _, _ -> "koblas-suitesparse has no bundled KLU for this host" }
