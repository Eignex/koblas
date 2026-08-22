package com.eignex.koblas.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions

/** LGPL-2.1-or-later KLU 2 extracted from Maven-native resources on the application's classpath. */
public class BundledKlu private constructor(private val delegate: KluSparseLu) : F64SparseLu by delegate {
    /** Extracts the matching Maven-native resources before the core FFM binding is initialized. */
    public constructor() : this(loadKlu())

    override val name: String get() = "klu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
}

private const val KLU_PATH_PROPERTY = "koblas.klu.path"
private const val KLU_PATH_ENVIRONMENT = "KOBLAS_KLU_PATH"

private fun loadKlu(): KluSparseLu {
    if (System.getProperty(KLU_PATH_PROPERTY) == null && System.getenv(KLU_PATH_ENVIRONMENT) == null) {
        System.setProperty(KLU_PATH_PROPERTY, KluResources.extract().toString())
    }
    return KluSparseLu()
}

internal object KluResources {
    private val x64Architectures = setOf("amd64", "x86_64")
    private val arm64Architectures = setOf("aarch64", "arm64")
    private val platform = when {
        System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
            System.getProperty("os.arch") in x64Architectures -> "linux-x86_64"

        System.getProperty("os.name").startsWith("Linux", ignoreCase = true) &&
            System.getProperty("os.arch") in arm64Architectures -> "linux-arm64"

        System.getProperty("os.name").startsWith("Mac", ignoreCase = true) &&
            System.getProperty("os.arch") in arm64Architectures -> "macosx-arm64"

        else -> error("koblas-klu has no bundled KLU for this host")
    }

    private val extracted: Path by lazy {
        val directory = Files.createTempDirectory("koblas-klu-$platform-")
        setOwnerOnlyPermissions(directory, "rwx------")
        resourceNames().forEach { name ->
            resource(name).use { input ->
                val destination = directory.resolve(name)
                Files.copy(input, destination)
                setOwnerOnlyPermissions(destination, "rw-------")
            }
        }
        directory.resolve(kluLibrary)
    }

    fun extract(): Path = extracted

    private val kluLibrary: String = if (platform.startsWith("linux")) "libklu.so.2" else "libklu.2.dylib"

    private fun resourceNames(): List<String> = checkNotNull(resource(".libraries")) {
        "KLU resources are absent for $platform"
    }.bufferedReader().useLines { it.filter(String::isNotBlank).toList() }

    private fun resource(name: String): InputStream? {
        val path = "org/eignex/klu/$platform/$name"
        return Thread.currentThread().contextClassLoader.getResourceAsStream(path)
            ?: BundledKlu::class.java.classLoader.getResourceAsStream(path)
            ?: BundledKlu::class.java.getResourceAsStream("/$path")
    }

    private fun setOwnerOnlyPermissions(path: Path, permissions: String) {
        runCatching { Files.setPosixFilePermissions(path, PosixFilePermissions.fromString(permissions)) }
            .onFailure { throw IllegalStateException("cannot secure extracted KLU resource $path", it) }
    }
}
