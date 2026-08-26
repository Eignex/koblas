package com.eignex.koblas.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu

/**
 * BASICLU from this module's bundled native resources, driven by koblas's own BASICLU binding. A deployment
 * pointing `koblas.basiclu.path` or `KOBLAS_BASICLU_PATH` at its own build gets that binding directly, and
 * discovery leaves this provider out of the running.
 */
public class BundledBasiclu private constructor(private val delegate: BasicluSparseLu) : F64SparseLu by delegate {
    /** Creates a BASICLU backend from the bundled library. */
    public constructor(factorizeMin: Int? = null, equilibrate: Boolean = false) :
        this(BasicluSparseLu(BasicluConfig(basicluLibrary.extract().toString(), factorizeMin, equilibrate)))

    override val name: String get() = "basiclu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 3

    /** BASICLU's own routines, forwarded so the bundled provider offers what a configured one does. */
    public val supportsBasisUpdates: Boolean get() = delegate.supportsBasisUpdates

    /** Factor a simplex [basis] for column replacements, which may bring in any column at all. */
    public fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization = delegate.factorBasis(basis)
}

private val basicluLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-basiclu",
    resourceRoot = "org/eignex/basiclu",
    anchor = BundledBasiclu::class.java,
    libraryDescription = "BASICLU",
    linuxSoname = "libkoblas_basiclu.so.1",
    macosSoname = "libkoblas_basiclu.1.dylib",
) { _, _ -> "koblas-basiclu has no bundled BASICLU for this host" }
