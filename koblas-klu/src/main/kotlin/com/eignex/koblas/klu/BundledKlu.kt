package com.eignex.koblas.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import java.nio.file.Path

/** LGPL-2.1-or-later KLU 2 extracted from Maven-native resources on the application's classpath. */
public class BundledKlu private constructor(private val delegate: KluSparseLu) : F64SparseLu by delegate {
    /** Extracts the matching Maven-native resources before the core FFM binding is initialized. */
    public constructor() : this(loadKlu())

    override val name: String get() = "klu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
}

private fun loadKlu(): KluSparseLu {
    return KluSparseLu(KluConfig(KluResources.extract().toString()))
}

internal object KluResources {
    private val platform = BundledNativeResources.supportedPlatform { _, _ ->
        "koblas-klu has no bundled KLU for this host"
    }
    private val resources = BundledNativeResources(
        directoryPrefix = "koblas-klu",
        platform = platform,
        resourceRoot = "org/eignex/klu",
        anchor = BundledKlu::class.java,
        libraryDescription = "KLU",
    )

    private val extracted: Path by lazy {
        checkNotNull(resources.extractRequired(resourceNames())[kluLibrary]) {
            "KLU resource is absent for $platform"
        }
    }

    fun extract(): Path = extracted

    private val kluLibrary: String = if (platform.startsWith("linux")) "libklu.so.2" else "libklu.2.dylib"

    private fun resourceNames(): List<String> = checkNotNull(resources.resource(".libraries")) {
        "KLU resources are absent for $platform"
    }.bufferedReader().useLines { it.filter(String::isNotBlank).toList() }
}
