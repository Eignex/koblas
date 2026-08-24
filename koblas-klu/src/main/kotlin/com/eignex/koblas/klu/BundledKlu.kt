package com.eignex.koblas.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu

/** KLU 2 bundle extracted from Maven-native resources on the application's classpath. */
public class BundledKlu private constructor(private val delegate: KluSparseLu) : F64SparseLu by delegate {
    /** Extracts the matching Maven-native resources before the core FFM binding is initialized. */
    public constructor() : this(loadKlu())

    override val name: String get() = "klu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
}

private fun loadKlu(): KluSparseLu = KluSparseLu(KluConfig(kluLibrary.extract().toString()))

private val kluLibrary = BundledNativeResources.manifestDriven(
    directoryPrefix = "koblas-klu",
    resourceRoot = "org/eignex/klu",
    anchor = BundledKlu::class.java,
    libraryDescription = "KLU",
    linuxSoname = "libklu.so.2",
    macosSoname = "libklu.2.dylib",
) { _, _ -> "koblas-klu has no bundled KLU for this host" }
