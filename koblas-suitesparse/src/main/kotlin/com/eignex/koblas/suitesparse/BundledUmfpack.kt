package com.eignex.koblas.suitesparse

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.openblas.BundledOpenBlas
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackOptions
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/**
 * SuiteSparse UMFPACK backend bundled in Maven-native resources.
 *
 * One of [UmfpackSparseLu] rather than a wrapper around one, so the bundled providers all answer to
 * [com.eignex.koblas.backendNamed] as the type their binding is.
 */
class BundledUmfpack private constructor(config: UmfpackConfig) : UmfpackSparseLu(config) {
    /** Creates bundled UMFPACK with default options. */
    constructor() : this(UmfpackOptions())

    /** Creates bundled UMFPACK with the same numerical and execution [options] accepted by the host binding. */
    constructor(options: UmfpackOptions) : this(bundledConfig(options))

    override val name: String get() = "umfpack-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
}

/** UMFPACK is linked against OpenBLAS, so the dependency is checked before the library is extracted. */
private fun bundledConfig(options: UmfpackOptions): UmfpackConfig {
    check(BundledOpenBlas().isAvailable) { "the bundled OpenBLAS dependency could not be loaded" }
    return UmfpackConfig(umfpackLibrary.extract().toString(), options)
}

private val umfpackLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-suitesparse",
    resourceRoot = "org/eignex/suitesparse",
    anchor = BundledUmfpack::class.java,
    libraryDescription = "UMFPACK",
    linuxSoname = "libumfpack.so.6",
    macosSoname = "libumfpack.dylib",
) { _, _ -> "koblas-suitesparse has no bundled UMFPACK for this host" }
