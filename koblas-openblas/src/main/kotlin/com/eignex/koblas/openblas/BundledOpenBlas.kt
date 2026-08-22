package com.eignex.koblas.openblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.hostblas.HostBlas
import com.eignex.koblas.hostblas.HostLapack
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions

/**
 * OpenBLAS and LAPACKE extracted from the Maven-native resources on this application's classpath.
 *
 * Add `com.eignex:koblas-openblas` at runtime to make this provider discoverable. The provider uses the
 * same FFM calls as the host backend, after putting the extracted absolute paths in its lookup properties.
 */
public class BundledOpenBlas private constructor(private val blas: HostBlas, private val lapack: HostLapack) :
    F64LinearAlgebra,
    F64Blas by blas,
    F64Lapack by lapack {

    /** Extracts the matching Maven-native resources before the core FFM binding is initialized. */
    public constructor() : this(loadHostBackends())

    private constructor(backends: HostBackends) : this(backends.blas, backends.lapack)

    override val name: String get() = "openblas-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
    override val isAvailable: Boolean get() = blas.isAvailable && lapack.isAvailable
    override val isPortable: Boolean get() = false
    override val vectorKernels: F64VectorKernels get() = blas.vectorKernels
}

private const val OPENBLAS_PATH_PROPERTY = "koblas.openblas.path"
private const val LAPACKE_PATH_PROPERTY = "koblas.lapacke.path"
private const val OPENBLAS_PATH_ENVIRONMENT = "KOBLAS_OPENBLAS_PATH"
private const val LAPACKE_PATH_ENVIRONMENT = "KOBLAS_LAPACKE_PATH"

internal data class OpenBlasPaths(val openblas: Path, val lapacke: Path?)

private data class HostBackends(val blas: HostBlas, val lapack: HostLapack)

private fun loadHostBackends(): HostBackends {
    if (openBlasPathOverride() == null) {
        configureHostLibraryPaths(OpenBlasResources.extract())
    }
    return HostBackends(HostBlas(), HostLapack())
}

internal fun configureHostLibraryPaths(paths: OpenBlasPaths) {
    setPathIfAbsent(OPENBLAS_PATH_PROPERTY, paths.openblas)
    paths.lapacke?.let { setPathIfAbsent(LAPACKE_PATH_PROPERTY, it) }
}

private fun setPathIfAbsent(property: String, path: Path) {
    val environment = when (property) {
        OPENBLAS_PATH_PROPERTY -> OPENBLAS_PATH_ENVIRONMENT
        LAPACKE_PATH_PROPERTY -> LAPACKE_PATH_ENVIRONMENT
        else -> error("unknown native library property $property")
    }
    if (System.getProperty(property) == null && System.getenv(environment) == null) {
        System.setProperty(property, path.toString())
    }
}

private fun openBlasPathOverride(): String? =
    System.getProperty(OPENBLAS_PATH_PROPERTY) ?: System.getenv(OPENBLAS_PATH_ENVIRONMENT)

internal object OpenBlasResources {
    private val x64Architectures: Set<String> = setOf("amd64", "x86_64")
    private val arm64Architectures: Set<String> = setOf("aarch64", "arm64")

    private val os: String = System.getProperty("os.name")
    private val architecture: String = System.getProperty("os.arch")

    private val platform: String = when {
        os.startsWith("Linux", ignoreCase = true) && architecture in x64Architectures -> "linux-x86_64"
        os.startsWith("Linux", ignoreCase = true) && architecture in arm64Architectures -> "linux-arm64"
        os.startsWith("Mac", ignoreCase = true) && architecture in arm64Architectures -> "macosx-arm64"
        else -> error("koblas-openblas has no bundled OpenBLAS for $os $architecture")
    }

    private val extracted: OpenBlasPaths by lazy {
        val directory = createExtractionDirectory()
        val copied = libraries.associateWith { name -> copyResource(directory, name) }
        OpenBlasPaths(
            checkNotNull(copied[openblasLibrary]) { "OpenBLAS resource is absent for $platform" },
            lapackeLibrary?.let(copied::get),
        )
    }

    fun extract(): OpenBlasPaths = extracted

    private fun createExtractionDirectory(): Path {
        val directory = Files.createTempDirectory("koblas-openblas-$platform-")
        setOwnerOnlyPermissions(directory, "rwx------")
        return directory
    }

    private fun copyResource(directory: Path, name: String): Path? {
        val input = resource(name) ?: return null
        val destination = directory.resolve(name)
        input.use { Files.copy(it, destination) }
        setOwnerOnlyPermissions(destination, "rw-------")
        return destination
    }

    private fun setOwnerOnlyPermissions(path: Path, permissions: String) {
        runCatching {
            Files.setPosixFilePermissions(path, PosixFilePermissions.fromString(permissions))
        }.onFailure { cause ->
            throw IllegalStateException("cannot secure extracted OpenBLAS resource $path", cause)
        }
    }

    private fun resource(name: String): InputStream? {
        val path = "org/bytedeco/openblas/$platform/$name"
        return Thread.currentThread().contextClassLoader.getResourceAsStream(path)
            ?: BundledOpenBlas::class.java.classLoader.getResourceAsStream(path)
            ?: BundledOpenBlas::class.java.getResourceAsStream("/$path")
    }

    private val openblasLibrary: String = when (platform) {
        "linux-x86_64", "linux-arm64" -> "libopenblas.so.0"
        "macosx-arm64" -> "libopenblas.0.dylib"
        else -> error("unsupported OpenBLAS platform $platform")
    }

    private val lapackeLibrary: String? = null

    private val libraries: List<String> = when (platform) {
        "linux-x86_64", "linux-arm64" -> listOf(
            openblasLibrary,
            "libgfortran.so.5",
            "libquadmath.so.0",
            "libgcc_s.so.1",
        )

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
