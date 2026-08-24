package com.eignex.koblas.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseLu

/** BASICLU backend bundled in Maven-native resources. */
public class BundledBasiclu private constructor(private val delegate: BasicluSparseLu) : F64SparseLu by delegate {
    /** Creates a BASICLU backend from bundled native resources. */
    public constructor(factorizeMin: Int? = null) :
        this(BasicluSparseLu(basicluLibrary.extract().toString(), factorizeMin))

    override val name: String get() = "basiclu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 3

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
