package com.eignex.koblas.hfactor

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64BasisSolvers
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu

/**
 * HiGHS's HFactor from this module's bundled native resources, driven by koblas's own HFactor binding. A
 * deployment pointing `koblas.hfactor.path` or `KOBLAS_HFACTOR_PATH` at its own build gets that binding
 * directly, and discovery leaves this provider out of the running.
 */
public class BundledHfactor private constructor(private val delegate: HfactorSparseLu) :
    F64SparseLu by delegate,
    F64BasisSolvers by delegate {
    /** Creates an HFactor backend from the bundled library. */
    public constructor(factorizeMin: Int? = null, equilibrate: Boolean = false) :
        this(HfactorSparseLu(HfactorConfig(hfactorLibrary.extract().toString(), factorizeMin, equilibrate)))

    override val name: String get() = "hfactor-bundled"

    // Both delegated halves are the same object, so either answers; Kotlin still wants the ambiguity settled.
    override val isPortable: Boolean get() = delegate.isPortable
    override val isAvailable: Boolean get() = delegate.isAvailable
    override val priority: Int get() = HOST_BACKEND_PRIORITY - 1

    override fun basisSolver(a: F64SparseMatrix): F64BasisSolver = delegate.basisSolver(a)
}

private val hfactorLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-hfactor",
    resourceRoot = "org/eignex/hfactor",
    anchor = BundledHfactor::class.java,
    libraryDescription = "HFactor",
    linuxSoname = "libkoblas_hfactor.so.1",
    macosSoname = "libkoblas_hfactor.1.dylib",
) { _, _ -> "koblas-hfactor has no bundled HFactor for this host" }
