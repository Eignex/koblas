package com.eignex.koblas.internal.backend

import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions

/** Shared extraction support for optional JVM artifacts that bundle native Maven resources. */
public class BundledNativeResources(
    private val directoryPrefix: String,
    private val platform: String,
    private val resourceRoot: String,
    private val anchor: Class<*>,
    private val libraryDescription: String,
) {
    /** Extracts every present resource into a private directory. */
    public fun extract(names: Iterable<String>): Map<String, Path> {
        val directory = Files.createTempDirectory("$directoryPrefix-$platform-")
        secure(directory, "rwx------")
        return names.mapNotNull { name ->
            resource(name)?.let { input ->
                val destination = directory.resolve(name)
                input.use { Files.copy(it, destination) }
                secure(destination, "rw-------")
                name to destination
            }
        }.toMap()
    }

    /** Extracts every named resource into a private directory. */
    public fun extractRequired(names: Iterable<String>): Map<String, Path> {
        val directory = Files.createTempDirectory("$directoryPrefix-$platform-")
        secure(directory, "rwx------")
        return names.associateWith { name ->
            val destination = directory.resolve(name)
            checkNotNull(resource(name)) { "$libraryDescription resource is absent for $platform" }.use {
                Files.copy(it, destination)
            }
            secure(destination, "rw-------")
            destination
        }
    }

    /** Opens a resource, preferring the context class loader. */
    public fun resource(name: String): InputStream? {
        val path = "$resourceRoot/$platform/$name"
        return Thread.currentThread().contextClassLoader.getResourceAsStream(path)
            ?: anchor.classLoader.getResourceAsStream(path)
            ?: anchor.getResourceAsStream("/$path")
    }

    private fun secure(path: Path, permissions: String) {
        runCatching { Files.setPosixFilePermissions(path, PosixFilePermissions.fromString(permissions)) }
            .onFailure { cause ->
                throw IllegalStateException("cannot secure extracted $libraryDescription resource $path", cause)
            }
    }

    /** Platform detection shared by bundled artifacts. */
    public companion object {
        /** Returns the Maven-native platform name supported by all bundled artifacts. */
        public fun supportedPlatform(unsupported: (os: String, architecture: String) -> String): String {
            val os = System.getProperty("os.name")
            val architecture = System.getProperty("os.arch")
            return when {
                os.startsWith("Linux", ignoreCase = true) && architecture in x64Architectures -> "linux-x86_64"
                os.startsWith("Linux", ignoreCase = true) && architecture in arm64Architectures -> "linux-arm64"
                os.startsWith("Mac", ignoreCase = true) && architecture in arm64Architectures -> "macosx-arm64"
                else -> error(unsupported(os, architecture))
            }
        }

        private val x64Architectures: Set<String> = setOf("amd64", "x86_64")
        private val arm64Architectures: Set<String> = setOf("aarch64", "arm64")
    }
}
