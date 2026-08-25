package com.eignex.koblas.openblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.*
import com.eignex.koblas.internal.backend.BundledNativeResources
import java.nio.file.Path

/** OpenBLAS backend bundled in Maven-native resources. */
public class BundledOpenBlas private constructor(private val blas: F64Cblas, private val decompositions: F64Lapacke) :
    F64LinearAlgebra,
    F64Blas by blas,
    F64Decompositions by decompositions {

    /** Creates an OpenBLAS backend from bundled native resources. */
    public constructor() : this(loadHostBackends())

    private constructor(backends: F64Backends) : this(backends.blas, backends.decompositions)

    override val name: String get() = "openblas-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    override val isAvailable: Boolean get() = blas.isAvailable && decompositions.isAvailable
    override val isPortable: Boolean get() = false
    override val kernels: F64Kernels get() = blas.kernels
}

internal data class OpenBlasPaths(val openblas: Path, val lapacke: Path?)

private fun loadHostBackends(): F64Backends {
    val paths = OpenBlasResources.extract()
    return F64Backends(HostBlasConfig(paths.openblas.toString(), paths.lapacke?.toString()))
}

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

    private val extracted: OpenBlasPaths by lazy {
        val copied = resources.extract(libraries)
        OpenBlasPaths(
            checkNotNull(copied[openblasLibrary]) { "OpenBLAS resource is absent for $platform" },
            lapackeLibrary?.let(copied::get),
        )
    }

    fun extract(): OpenBlasPaths = extracted

    private val openblasLibrary: String = when (platform) {
        "linux-x86_64", "linux-arm64" -> "libopenblas.so.0"
        "macosx-arm64" -> "libopenblas.0.dylib"
        else -> error("unsupported OpenBLAS platform $platform")
    }

    private val lapackeLibrary: String? = null

    private val libraries: List<String> = when (platform) {
        "linux-x86_64" -> listOf(
            openblasLibrary,
            "libgfortran.so.5",
            "libquadmath.so.0",
            "libgcc_s.so.1",
        )

        "linux-arm64" -> listOf(openblasLibrary, "libgfortran.so.5", "libgcc_s.so.1")

        "macosx-arm64" -> listOf(
            openblasLibrary,
            "libgfortran.dylib",
            "libgfortran.5.dylib",
            "libquadmath.0.dylib",
            "libgcc_s.1.1.dylib",
        )

        else -> error("unsupported OpenBLAS platform $platform")
    }
}
