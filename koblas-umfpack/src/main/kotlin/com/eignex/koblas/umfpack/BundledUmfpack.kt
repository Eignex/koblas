package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.openblas.BundledOpenBlas
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu
import java.nio.file.Path

/**
 * GPL-3.0-only SuiteSparse UMFPACK extracted from Maven-native resources on this application's classpath.
 *
 * Add `com.eignex:koblas-umfpack` at runtime to discover this optional sparse backend. Its GPL license is
 * separate from the Apache-2.0 [koblas](https://github.com/Eignex/koblas) core artifact.
 */
public class BundledUmfpack private constructor(private val delegate: UmfpackSparseLu) : F64SparseLu by delegate {
    /** Extracts the matching Maven-native resources before the core FFM binding is initialized. */
    public constructor() : this(loadUmfpack())

    override val name: String get() = "umfpack-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
}

private fun loadUmfpack(): UmfpackSparseLu {
    // UMFPACK links to BLAS. Loading the transitive single OpenBLAS bundle first makes its SONAME available
    // to the dynamic loader without making a system installation part of this optional artifact's contract.
    check(BundledOpenBlas().isAvailable) { "the bundled OpenBLAS dependency could not be loaded" }
    return UmfpackSparseLu(UmfpackConfig(libraryPath = UmfpackResources.extract().toString()))
}

internal object UmfpackResources {
    private val platform = BundledNativeResources.supportedPlatform { _, _ ->
        "koblas-umfpack has no bundled SuiteSparse for this host"
    }
    private val resources = BundledNativeResources(
        directoryPrefix = "koblas-umfpack",
        platform = platform,
        resourceRoot = "org/eignex/umfpack",
        anchor = BundledUmfpack::class.java,
        libraryDescription = "SuiteSparse",
    )

    private val extracted: Path by lazy {
        checkNotNull(resources.extractRequired(resourceNames())[umfpackLibrary]) {
            "UMFPACK resource is absent for $platform"
        }
    }

    fun extract(): Path = extracted

    private val umfpackLibrary: String = if (platform.startsWith("linux")) "libumfpack.so.6" else "libumfpack.dylib"

    private fun resourceNames(): List<String> = checkNotNull(resources.resource(".libraries")) {
        "UMFPACK resources are absent for $platform"
    }
        .bufferedReader().useLines { it.filter(String::isNotBlank).toList() }
}
