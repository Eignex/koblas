package com.eignex.koblas.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu

/** KLU backend bundled in Maven-native resources. */
public class BundledKlu private constructor(private val delegate: KluSparseLu) : F64SparseLu by delegate {
    /** Creates a KLU backend from bundled native resources. */
    public constructor(factorizeMin: Int? = null) : this(loadKlu(factorizeMin))

    override val name: String get() = "klu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
}

private fun loadKlu(factorizeMin: Int?): KluSparseLu =
    KluSparseLu(KluConfig(kluLibrary.extract().toString(), factorizeMin))

private val kluLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-klu",
    resourceRoot = "org/eignex/klu",
    anchor = BundledKlu::class.java,
    libraryDescription = "KLU",
    linuxSoname = "libklu.so.2",
    macosSoname = "libklu.2.dylib",
) { _, _ -> "koblas-klu has no bundled KLU for this host" }
