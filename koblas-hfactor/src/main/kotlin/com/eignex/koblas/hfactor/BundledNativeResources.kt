package com.eignex.koblas.hfactor

import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap

internal class BundledNativeResources(
    private val directoryPrefix: String,
    private val platform: String,
    private val resourceRoot: String,
    private val anchor: Class<*>,
    private val libraryDescription: String,
) {
    private val key: String get() = "$directoryPrefix|$resourceRoot|$platform"

    @Suppress("TooGenericExceptionCaught")
    private fun extractManifest(): Map<String, Path> {
        val pending = CompletableFuture<Map<String, Path>>()
        val existing = extractions.putIfAbsent(key, pending)
        if (existing != null) return existing.join()
        try {
            return extract(manifestNames(), required = true).also(pending::complete)
        } catch (cause: Throwable) {
            pending.completeExceptionally(cause)
            extractions.remove(key, pending)
            throw cause
        }
    }

    private fun extract(names: Iterable<String>, required: Boolean = false): Map<String, Path> {
        val directory = Files.createTempDirectory("$directoryPrefix-$platform-")
        secure(directory, "rwx------")
        return names.mapNotNull { name ->
            val input = resource(name)
            check(input != null || !required) { "$libraryDescription resource is absent for $platform" }
            input?.let {
                val destination = directory.resolve(name)
                it.use { stream -> Files.copy(stream, destination) }
                secure(destination, "rw-------")
                name to destination
            }
        }.toMap()
    }

    private fun manifestNames(): List<String> = checkNotNull(resource(".libraries")) {
        "$libraryDescription resources are absent for $platform"
    }.bufferedReader().useLines { lines -> lines.filter(String::isNotBlank).toList() }

    private fun resource(name: String): InputStream? {
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

    internal companion object {
        private val extractions = ConcurrentHashMap<String, CompletableFuture<Map<String, Path>>>()

        @Suppress("LongParameterList")
        fun manifestDriven(
            directoryPrefix: String,
            resourceRoot: String,
            anchor: Class<*>,
            libraryDescription: String,
            linuxSoname: String,
            macosSoname: String,
            unsupported: (os: String, architecture: String) -> String,
        ): BundledLibrary {
            val platform = supportedPlatform(unsupported)
            val resources = BundledNativeResources(
                directoryPrefix,
                platform,
                resourceRoot,
                anchor,
                libraryDescription,
            )
            val soname = if (platform.startsWith("linux")) linuxSoname else macosSoname
            return BundledLibrary(resources, soname, libraryDescription, platform)
        }

        private fun supportedPlatform(unsupported: (os: String, architecture: String) -> String): String {
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

    internal class BundledLibrary(
        private val resources: BundledNativeResources,
        private val soname: String,
        private val description: String,
        private val platform: String,
    ) {
        private val extracted: Path by lazy {
            checkNotNull(resources.extractManifest()[soname]) {
                "$description resource $soname is absent for $platform"
            }
        }

        fun extract(): Path = extracted
    }
}
