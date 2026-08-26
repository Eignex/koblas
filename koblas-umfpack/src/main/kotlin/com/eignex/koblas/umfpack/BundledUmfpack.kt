package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.openblas.BundledOpenBlas
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/** SuiteSparse UMFPACK backend bundled in Maven-native resources. */
public class BundledUmfpack private constructor(private val delegate: UmfpackSparseLu) : F64SparseLu by delegate {
    /** Creates a UMFPACK backend from bundled native resources. */
    public constructor(factorizeMin: Int? = null, equilibrate: Boolean = false) :
        this(loadUmfpack(factorizeMin, equilibrate))

    override val name: String get() = "umfpack-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
}

private fun loadUmfpack(factorizeMin: Int?, equilibrate: Boolean): UmfpackSparseLu {
    check(BundledOpenBlas().isAvailable) { "the bundled OpenBLAS dependency could not be loaded" }
    return UmfpackSparseLu(
        UmfpackConfig(
            libraryPath = umfpackLibrary.extract().toString(),
            factorizeMin = factorizeMin,
            equilibrate = equilibrate,
        ),
    )
}

private val umfpackLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-umfpack",
    resourceRoot = "org/eignex/umfpack",
    anchor = BundledUmfpack::class.java,
    libraryDescription = "SuiteSparse",
    linuxSoname = "libumfpack.so.6",
    macosSoname = "libumfpack.dylib",
) { _, _ -> "koblas-umfpack has no bundled SuiteSparse for this host" }
