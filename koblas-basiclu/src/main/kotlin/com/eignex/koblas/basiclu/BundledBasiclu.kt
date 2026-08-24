package com.eignex.koblas.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseLu

/** BASICLU bundle extracted from Maven-native resources on the application's classpath. */
public class BundledBasiclu private constructor(private val delegate: BasicluSparseLu) : F64SparseLu by delegate {
    /**
     * Extracts the matching Maven-native resources before the FFM binding is initialized.
     *
     * @param factorizeMin stored entries from which this binding factorizes natively, or null for the
     *   platform default.
     */
    public constructor(factorizeMin: Int? = null) :
        this(BasicluSparseLu(basicluLibrary.extract().toString(), factorizeMin))

    override val name: String get() = "basiclu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 3

    /** Factor a simplex [basis] for sparse column replacements. */
    override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization = delegate.factorBasis(basis)
}

private val basicluLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-basiclu",
    resourceRoot = "org/eignex/basiclu",
    anchor = BundledBasiclu::class.java,
    libraryDescription = "BASICLU",
    linuxSoname = "libkoblas_basiclu.so.1",
    macosSoname = "libkoblas_basiclu.1.dylib",
) { _, _ -> "koblas-basiclu has no bundled BASICLU for this host" }
