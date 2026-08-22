package com.eignex.koblas.superlu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.openblas.BundledOpenBlas
import com.eignex.koblas.sparse.F64SparseLapack
import com.eignex.koblas.sparse.host.superlu.SuperluDriver
import com.eignex.koblas.sparse.host.superlu.SuperluSparseLapack
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions

/** BSD-3-Clause SuperLU 7 extracted from Maven-native resources on the application's classpath. */
public class BundledSuperlu private constructor(private val delegate: SuperluSparseLapack) :
    F64SparseLapack by delegate {
    /** Extracts the matching Maven-native resources before the core FFM binding is initialized. */
    public constructor(driver: SuperluDriver = SuperluDriver.Expert) : this(loadSuperlu(driver))

    override val name: String get() = "superlu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
}

private const val SUPERLU_PATH_PROPERTY = "koblas.superlu.path"
private const val SUPERLU_PATH_ENVIRONMENT = "KOBLAS_SUPERLU_PATH"

private fun loadSuperlu(driver: SuperluDriver): SuperluSparseLapack {
    check(BundledOpenBlas().isAvailable) { "the bundled OpenBLAS dependency could not be loaded" }
    if (System.getProperty(SUPERLU_PATH_PROPERTY) == null &&
        System.getenv(SUPERLU_PATH_ENVIRONMENT) == null
    ) {
        System.setProperty(SUPERLU_PATH_PROPERTY, SuperluResources.extract().toString())
    }
    return SuperluSparseLapack(driver)
}

internal object SuperluResources {
    private val x64Architectures = setOf("amd64", "x86_64")
    private val arm64Architectures = setOf("aarch64", "arm64")
    private val platform = when {
        System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
            System.getProperty("os.arch") in x64Architectures -> "linux-x86_64"

        System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
            System.getProperty("os.arch") in arm64Architectures -> "linux-arm64"

        System.getProperty("os.name").startsWith("Mac", ignoreCase = true) &&
            System.getProperty("os.arch") in arm64Architectures -> "macosx-arm64"

        else -> error("koblas-superlu has no bundled SuperLU for this host")
    }

    private val extracted: Path by lazy {
        val directory = Files.createTempDirectory("koblas-superlu-$platform-")
        setOwnerOnlyPermissions(directory, "rwx------")
        resourceNames().forEach { name ->
            resource(name).use { input ->
                val destination = directory.resolve(name)
                Files.copy(input, destination)
                setOwnerOnlyPermissions(destination, "rw-------")
            }
        }
        directory.resolve(superluLibrary)
    }

    fun extract(): Path = extracted

    private val superluLibrary: String = if (platform.startsWith("linux")) "libsuperlu.so.7" else "libsuperlu.7.dylib"

    private fun resourceNames(): List<String> = checkNotNull(resource(".libraries")) {
        "SuperLU resources are absent for $platform"
    }.bufferedReader().useLines { it.filter(String::isNotBlank).toList() }

    private fun resource(name: String): InputStream? {
        val path = "org/eignex/superlu/$platform/$name"
        return Thread.currentThread().contextClassLoader.getResourceAsStream(path)
            ?: BundledSuperlu::class.java.classLoader.getResourceAsStream(path)
            ?: BundledSuperlu::class.java.getResourceAsStream("/$path")
    }

    private fun setOwnerOnlyPermissions(path: Path, permissions: String) {
        runCatching { Files.setPosixFilePermissions(path, PosixFilePermissions.fromString(permissions)) }
            .onFailure {
                throw IllegalStateException("cannot secure extracted SuperLU resource $path", it)
            }
    }
}
