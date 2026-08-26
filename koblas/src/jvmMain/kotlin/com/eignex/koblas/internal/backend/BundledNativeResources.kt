package com.eignex.koblas.internal.backend

import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions
import java.util.concurrent.ConcurrentHashMap

/** Shared extraction support for optional JVM artifacts that bundle native Maven resources. */
public class BundledNativeResources(
    private val directoryPrefix: String,
    private val platform: String,
    private val resourceRoot: String,
    private val anchor: Class<*>,
    private val libraryDescription: String,
) {
    /** What identifies one bundle, so two libraries out of the same one share a single extraction. */
    private val key: String get() = "$directoryPrefix|$resourceRoot|$platform"

    /**
     * The whole manifest, extracted once per bundle per process. An artifact bundling several libraries has
     * a handle per library and each asks for the whole manifest, since a library's dependencies ship beside
     * it; without this they would each lay the bundle down in a directory of their own.
     */
    internal fun extractManifest(): Map<String, Path> =
        extractions.getOrPut(key) { extract(manifestNames(), required = true) }

    /**
     * Extracts [names] into a private directory. A name this bundle does not carry is skipped, or fails when
     * [required], which is what a manifest listing exactly what shipped calls for.
     */
    public fun extract(names: Iterable<String>, required: Boolean = false): Map<String, Path> {
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

    /** The names in this bundle's `.libraries` manifest, which lists what shipped for this platform. */
    public fun manifestNames(): List<String> = checkNotNull(resource(".libraries")) {
        "$libraryDescription resources are absent for $platform"
    }.bufferedReader().useLines { lines -> lines.filter(String::isNotBlank).toList() }

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

    /** Platform detection and the manifest-driven bundle shared by bundled artifacts. */
    public companion object {
        /**
         * One extraction per bundle, keyed by what identifies it. Held for the life of the process because
         * the paths it hands out stay loaded, and shared across threads because two backends out of one
         * artifact may be constructed at once.
         */
        private val extractions = ConcurrentHashMap<String, Map<String, Path>>()

        /**
         * A bundle whose contents are listed by a `.libraries` manifest and whose one library of interest is
         * [linuxSoname] or [macosSoname]. Every optional artifact koblas ships that wraps a single library is
         * this, so each contributes only its names.
         */
        @Suppress("LongParameterList") // one parameter per thing that differs between the bundles
        public fun manifestDriven(
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
                directoryPrefix = directoryPrefix,
                platform = platform,
                resourceRoot = resourceRoot,
                anchor = anchor,
                libraryDescription = libraryDescription,
            )
            val soname = if (platform.startsWith("linux")) linuxSoname else macosSoname
            return BundledLibrary(resources, soname, libraryDescription, platform)
        }

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

/**
 * One bundled native library, extracted from the classpath on first use.
 *
 * The whole manifest is extracted, since a library's dependencies ship beside it and the loader needs them
 * in the same directory, and [extract] hands back the path of the library itself.
 */
public class BundledLibrary internal constructor(
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

    /** The extracted library's absolute path, extracting the bundle the first time it is asked for. */
    public fun extract(): Path = extracted
}
