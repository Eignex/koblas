package com.eignex.koblas.suitesparse

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.openblas.BundledOpenBlas
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackOptions
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/**
 * SuiteSparse UMFPACK backend bundled in Maven-native resources.
 *
 * One of [UmfpackSparseLu] rather than a wrapper around one, so the bundled providers all answer to
 * [com.eignex.koblas.sparseDecompositionsNamed] as the type their binding is.
 */
public class BundledUmfpack private constructor(config: UmfpackConfig) : UmfpackSparseLu(config) {
    /** Creates bundled UMFPACK with default options. */
    public constructor() : this(UmfpackOptions())

    /** Creates bundled UMFPACK with the same numerical and execution [options] accepted by the host binding. */
    public constructor(options: UmfpackOptions) : this(bundledConfig(options))

    /** Retains the original threshold-and-equilibration constructor. */
    public constructor(factorizeMin: Int?, equilibrate: Boolean = false) : this(
        UmfpackOptions(factorizeMin = factorizeMin, equilibrate = equilibrate),
    )

    /** Retains calls that configured only equilibration by name. */
    public constructor(equilibrate: Boolean) : this(UmfpackOptions(equilibrate = equilibrate))

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
