package com.eignex.koblas.internal.kernels

import com.eignex.koblas.internal.host.FfmLibrary
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions

/** FFM bindings to the C kernels bundled in the JVM artifact. */
internal object JvmCKernelBindings {
    private val symbolNames = listOf(
        "koblas_dense_dot",
        "koblas_dense_ssqd",
        "koblas_dense_nrm2",
        "koblas_dense_sum",
        "koblas_dense_asum",
        "koblas_dense_dot4",
        "koblas_sparse_dot_dense",
    )
    private val library: FfmLibrary? = loadLibraryOrNull()

    val isAvailable: Boolean get() = library != null

    private fun requiredLibrary(): FfmLibrary = checkNotNull(library) { "bundled koblas C kernels are unavailable" }

    private val denseDot by lazy {
        requiredLibrary().handle(
            "koblas_dense_dot",
            FfmLibrary.doubleOf(ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT),
        )
    }
    private val denseSsqd by lazy {
        requiredLibrary().handle(
            "koblas_dense_ssqd",
            FfmLibrary.doubleOf(ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT),
        )
    }
    private val denseNrm2 by lazy {
        requiredLibrary().handle(
            "koblas_dense_nrm2",
            FfmLibrary.doubleOf(ADDRESS, JAVA_INT, JAVA_INT),
        )
    }
    private val denseSum by lazy {
        requiredLibrary().handle("koblas_dense_sum", FfmLibrary.doubleOf(ADDRESS, JAVA_INT, JAVA_INT))
    }
    private val denseAsum by lazy {
        requiredLibrary().handle(
            "koblas_dense_asum",
            FfmLibrary.doubleOf(ADDRESS, JAVA_INT, JAVA_INT),
        )
    }
    private val denseDot4 by lazy {
        requiredLibrary().handle(
            "koblas_dense_dot4",
            FfmLibrary.voidOf(ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }
    private val sparseDotDense by lazy {
        requiredLibrary().handle(
            "koblas_sparse_dot_dense",
            FfmLibrary.doubleOf(ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
        )
    }

    fun denseDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        denseDot.invokeExact(MemorySegment.ofArray(a), aOff, MemorySegment.ofArray(b), bOff, len) as Double

    fun denseSsqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = denseSsqd.invokeExact(
        MemorySegment.ofArray(a),
        aOff,
        MemorySegment.ofArray(b),
        bOff,
        len,
    ) as Double

    fun denseNrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        denseNrm2.invokeExact(MemorySegment.ofArray(v), vOff, len) as Double

    fun denseSum(v: DoubleArray, vOff: Int, len: Int): Double =
        denseSum.invokeExact(MemorySegment.ofArray(v), vOff, len) as Double

    fun denseAsum(v: DoubleArray, vOff: Int, len: Int): Double =
        denseAsum.invokeExact(MemorySegment.ofArray(v), vOff, len) as Double

    @Suppress("LongParameterList")
    fun denseDot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        denseDot4.invokeExact(
            MemorySegment.ofArray(a),
            aOff,
            stride,
            MemorySegment.ofArray(b),
            bOff,
            len,
            MemorySegment.ofArray(out),
            outOff,
        ) as Unit
    }

    @Suppress("LongParameterList")
    fun sparseDotDense(indices: IntArray, values: DoubleArray, dense: DoubleArray): Double = sparseDotDense.invokeExact(
        MemorySegment.ofArray(indices),
        MemorySegment.ofArray(values),
        indices.size,
        MemorySegment.ofArray(dense),
    ) as Double

    private fun loadLibraryOrNull(): FfmLibrary? = try {
        val extractedLibrary = extractLibrary()
        FfmLibrary.open(
            listOf(extractedLibrary.toString()),
            "koblas_dense_dot",
            "bundled koblas C kernels",
        ).takeIf { it.containsAll(symbolNames) }
    } catch (_: RuntimeException) {
        null
    } catch (_: UnsatisfiedLinkError) {
        null
    }

    private fun extractLibrary(): Path {
        val (platform, libraryName) = supportedPlatform()
        val resource = "com/eignex/koblas/internal/kernels/$platform/$libraryName"
        val stream = Thread.currentThread().contextClassLoader.getResourceAsStream(resource)
            ?: JvmCKernelBindings::class.java.classLoader.getResourceAsStream(resource)
            ?: error("bundled C kernel resource is absent for $platform")
        val directory = Files.createTempDirectory("koblas-kernels-$platform-")
        secure(directory, "rwx------")
        val destination = directory.resolve(libraryName)
        stream.use { Files.copy(it, destination) }
        secure(destination, "rw-------")
        destination.toFile().deleteOnExit()
        directory.toFile().deleteOnExit()
        return destination
    }

    private fun secure(path: Path, permissions: String) {
        runCatching {
            Files.setPosixFilePermissions(path, PosixFilePermissions.fromString(permissions))
        }
            .getOrElse { cause ->
                throw IllegalStateException(
                    "cannot secure extracted C kernel resource $path",
                    cause,
                )
            }
    }

    private fun supportedPlatform(): Pair<String, String> {
        val os = System.getProperty("os.name")
        val architecture = System.getProperty("os.arch")
        return when {
            os.startsWith("Linux", ignoreCase = true) && architecture in setOf("amd64", "x86_64") ->
                "linux-x86_64" to "libkoblas_kernels.so"

            os.startsWith("Linux", ignoreCase = true) && architecture in setOf("aarch64", "arm64") ->
                "linux-arm64" to "libkoblas_kernels.so"

            os.startsWith("Mac", ignoreCase = true) && architecture in setOf("aarch64", "arm64") ->
                "macosx-arm64" to "libkoblas_kernels.dylib"

            else -> error("unsupported koblas C kernel host $os/$architecture")
        }
    }
}
