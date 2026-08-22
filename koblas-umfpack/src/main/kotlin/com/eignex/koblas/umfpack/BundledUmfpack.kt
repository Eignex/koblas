package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.openblas.BundledOpenBlas
import com.eignex.koblas.sparse.F64SparseLapack
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLapack
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions

/**
 * GPL-3.0-only SuiteSparse UMFPACK extracted from Maven-native resources on this application's classpath.
 *
 * Add `com.eignex:koblas-umfpack` at runtime to discover this optional sparse backend. Its GPL license is
 * separate from the Apache-2.0 [koblas](https://github.com/Eignex/koblas) core artifact.
 */
public class BundledUmfpack private constructor(private val delegate: UmfpackSparseLapack) :
    F64SparseLapack by delegate {
    /** Extracts the matching Maven-native resources before the core FFM binding is initialized. */
    public constructor() : this(loadUmfpack())

    override val name: String get() = "umfpack-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 1
}

private const val UMFPACK_PATH_PROPERTY = "koblas.umfpack.path"
private const val UMFPACK_PATH_ENVIRONMENT = "KOBLAS_UMFPACK_PATH"

private fun loadUmfpack(): UmfpackSparseLapack {
    // UMFPACK links to BLAS. Loading the transitive single OpenBLAS bundle first makes its SONAME available
    // to the dynamic loader without making a system installation part of this optional artifact's contract.
    check(BundledOpenBlas().isAvailable) { "the bundled OpenBLAS dependency could not be loaded" }
    if (System.getProperty(UMFPACK_PATH_PROPERTY) == null && System.getenv(UMFPACK_PATH_ENVIRONMENT) == null) {
        System.setProperty(UMFPACK_PATH_PROPERTY, UmfpackResources.extract().toString())
    }
    return UmfpackSparseLapack()
}

internal object UmfpackResources {
    private val x64Architectures = setOf("amd64", "x86_64")
    private val arm64Architectures = setOf("aarch64", "arm64")
    private val platform = when {
        System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
            System.getProperty("os.arch") in x64Architectures -> "linux-x86_64"

        System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
            System.getProperty("os.arch") in arm64Architectures -> "linux-arm64"

        System.getProperty("os.name").startsWith("Mac", ignoreCase = true) &&
            System.getProperty("os.arch") in arm64Architectures -> "macosx-arm64"

        else -> error("koblas-umfpack has no bundled SuiteSparse for this host")
    }

    private val extracted: Path by lazy {
        val directory = Files.createTempDirectory("koblas-umfpack-$platform-")
        setOwnerOnlyPermissions(directory, "rwx------")
        resourceNames().forEach { name ->
            resource(name).use { input ->
                val destination = directory.resolve(name)
                Files.copy(input, destination)
                setOwnerOnlyPermissions(destination, "rw-------")
            }
        }
        directory.resolve(umfpackLibrary)
    }

    fun extract(): Path = extracted

    private val umfpackLibrary: String = if (platform.startsWith("linux")) "libumfpack.so.6" else "libumfpack.dylib"

    private fun resourceNames(): List<String> = checkNotNull(resource(".libraries")) {
        "UMFPACK resources are absent for $platform"
    }
        .bufferedReader().useLines { it.filter(String::isNotBlank).toList() }

    private fun resource(name: String): InputStream? {
        val path = "org/eignex/umfpack/$platform/$name"
        return Thread.currentThread().contextClassLoader.getResourceAsStream(path)
            ?: BundledUmfpack::class.java.classLoader.getResourceAsStream(path)
            ?: BundledUmfpack::class.java.getResourceAsStream("/$path")
    }

    private fun setOwnerOnlyPermissions(path: Path, permissions: String) {
        runCatching { Files.setPosixFilePermissions(path, PosixFilePermissions.fromString(permissions)) }
            .onFailure { throw IllegalStateException("cannot secure extracted SuiteSparse resource $path", it) }
    }
}
