package com.eignex.koblas.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.internal.backend.configuredBasicluPath
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseLu

/**
 * BASICLU backend, from the library a deployment configured through `koblas.basiclu.path` or
 * `KOBLAS_BASICLU_PATH` when it named one, and from the bundled native resources otherwise.
 */
public class BundledBasiclu private constructor(
    private val delegate: BasicluSparseLu,
    private val configured: Boolean,
) : F64SparseLu by delegate {
    /** Creates a BASICLU backend, preferring a configured library over the bundled one. */
    public constructor(factorizeMin: Int? = null) : this(configuredBasicluPath(), factorizeMin)

    private constructor(configuredPath: String?, factorizeMin: Int?) : this(
        BasicluSparseLu(configuredPath ?: basicluLibrary.extract().toString(), factorizeMin),
        configuredPath != null,
    )

    /** The suffix marks the bundled build, so a configured library reports the canonical name instead. */
    override val name: String get() = if (configured) "basiclu" else "basiclu-bundled"

    override val priority: Int get() = HOST_BACKEND_PRIORITY + 3

    override val supportsBasisUpdates: Boolean get() = delegate.supportsBasisUpdates

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
