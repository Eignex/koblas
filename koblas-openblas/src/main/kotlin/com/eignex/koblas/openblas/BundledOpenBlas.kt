package com.eignex.koblas.openblas

import com.eignex.koblas.*
import com.eignex.koblas.F64BundledBackend
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Kernels
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.cblas.OpenBlasOptions
import com.eignex.koblas.dense.host.jvm.*
import java.nio.file.Path

/** CBLAS backend bundled in Maven-native resources. */
class BundledOpenBlas private constructor(private val blas: F64Cblas) :
    F64BundledBackend,
    Blas by blas,
    F64RoutingBackend,
    BackendMetadataProvider {

    /** Creates an OpenBLAS backend from bundled native resources with default options. */
    constructor() : this(OpenBlasOptions())

    /** Creates an OpenBLAS backend from bundled native resources with shared [options]. */
    constructor(options: OpenBlasOptions) : this(loadHostBackend(options))

    override val name: String get() = "openblas-bundled"

    /** The name a deployment configures this library under, whichever build provides it. */
    override val canonicalName: String get() = "openblas"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    override val isAvailable: Boolean get() = blas.isAvailable
    override val unavailableReason: String? get() = blas.unavailableReason
    override val isPortable: Boolean get() = false
    override val kernels: Kernels get() = blas.kernels
    override val backendMetadata: BackendMetadata get() = blas.backendMetadata

    override fun route(query: F64RouteQuery): BackendRoute? = blas.route(query)
        ?.let { if (it.execution == BackendExecution.NATIVE) it.copy(executor = name) else it }
}

private fun loadHostBackend(options: OpenBlasOptions): F64Cblas =
    F64Cblas(HostBlasConfig(OpenBlasResources.extract().toString(), options))

internal object OpenBlasResources {
    private val platform: String = BundledNativeResources.supportedPlatform { os, architecture ->
        "koblas-openblas has no bundled OpenBLAS for $os $architecture"
    }
    private val resources = BundledNativeResources(
        directoryPrefix = "koblas-openblas",
        platform = platform,
        resourceRoot = "org/bytedeco/openblas",
        anchor = BundledOpenBlas::class.java,
        libraryDescription = "OpenBLAS",
    )

    private val extracted: Path by lazy {
        val copied = resources.extract(libraries)
        checkNotNull(copied[openblasLibrary]) { "OpenBLAS resource is absent for $platform" }
    }

    fun extract(): Path = extracted

    private val openblasLibrary: String = when (platform) {
        "linux-x86_64", "linux-arm64" -> "libopenblas.so.0"
        "macosx-arm64" -> "libopenblas.0.dylib"
        else -> error("unsupported OpenBLAS platform $platform")
    }

    private val libraries: List<String> = listOf(openblasLibrary)
}
