package com.eignex.koblas.internal.kernels

import com.eignex.koblas.internal.host.FfmLibrary
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions

/** FFM bindings to the C kernels bundled in the JVM artifact. */
internal object JvmCKernelBindings {
    private val symbolNames = listOf(
        "koblas_dense_dot",
        "koblas_dense_ssqd",
        "koblas_dense_axpy",
        "koblas_dense_axpy_arithmetic",
        "koblas_dense_scale",
        "koblas_dense_nrm2",
        "koblas_dense_sum",
        "koblas_dense_asum",
        "koblas_dense_swap",
        "koblas_dense_dot4",
        "koblas_dense_rotm",
        "koblas_sparse_dot_dense",
        "koblas_sparse_dot_sparse",
        "koblas_sparse_axpy",
        "koblas_sparse_scatter",
        "koblas_sparse_gather",
        "koblas_sparse_gather_zero",
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
    private val denseAxpy by lazy {
        requiredLibrary().handle(
            "koblas_dense_axpy",
            FfmLibrary.voidOf(ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, JAVA_INT),
        )
    }
    private val denseAxpyArithmetic by lazy {
        requiredLibrary().handle(
            "koblas_dense_axpy_arithmetic",
            FfmLibrary.voidOf(ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, JAVA_INT),
        )
    }
    private val denseScale by lazy {
        requiredLibrary().handle(
            "koblas_dense_scale",
            FfmLibrary.voidOf(ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_INT),
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
    private val denseSwap by lazy {
        requiredLibrary().handle(
            "koblas_dense_swap",
            FfmLibrary.voidOf(ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT),
        )
    }
    private val denseDot4 by lazy {
        requiredLibrary().handle(
            "koblas_dense_dot4",
            FfmLibrary.voidOf(ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }
    private val denseRotm by lazy {
        requiredLibrary().handle(
            "koblas_dense_rotm",
            FfmLibrary.voidOf(
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
                JAVA_DOUBLE,
            ),
        )
    }
    private val sparseDotDense by lazy {
        requiredLibrary().handle(
            "koblas_sparse_dot_dense",
            FfmLibrary.doubleOf(ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
        )
    }
    private val sparseDotSparse by lazy {
        requiredLibrary().handle(
            "koblas_sparse_dot_sparse",
            FfmLibrary.doubleOf(ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT),
        )
    }
    private val sparseAxpy by lazy {
        requiredLibrary().handle(
            "koblas_sparse_axpy",
            FfmLibrary.voidOf(ADDRESS, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS),
        )
    }
    private val sparseScatter by lazy {
        requiredLibrary().handle(
            "koblas_sparse_scatter",
            FfmLibrary.voidOf(ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
        )
    }
    private val sparseGather by lazy {
        requiredLibrary().handle(
            "koblas_sparse_gather",
            FfmLibrary.voidOf(ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
        )
    }
    private val sparseGatherZero by lazy {
        requiredLibrary().handle(
            "koblas_sparse_gather_zero",
            FfmLibrary.voidOf(ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
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

    fun denseAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        denseAxpy.invokeExact(
            MemorySegment.ofArray(y),
            yOff,
            alpha,
            MemorySegment.ofArray(x),
            xOff,
            len,
        ) as Unit
    }

    fun denseAxpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        denseAxpyArithmetic.invokeExact(
            MemorySegment.ofArray(y),
            yOff,
            alpha,
            MemorySegment.ofArray(x),
            xOff,
            len,
        ) as Unit
    }

    fun denseScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        denseScale.invokeExact(MemorySegment.ofArray(v), vOff, alpha, len) as Unit
    }

    fun denseNrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        denseNrm2.invokeExact(MemorySegment.ofArray(v), vOff, len) as Double

    fun denseSum(v: DoubleArray, vOff: Int, len: Int): Double =
        denseSum.invokeExact(MemorySegment.ofArray(v), vOff, len) as Double

    fun denseAsum(v: DoubleArray, vOff: Int, len: Int): Double =
        denseAsum.invokeExact(MemorySegment.ofArray(v), vOff, len) as Double

    fun denseSwap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        denseSwap.invokeExact(MemorySegment.ofArray(a), aOff, MemorySegment.ofArray(b), bOff, len) as Unit
    }

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
    fun denseRotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        h11: Double,
        h12: Double,
        h21: Double,
        h22: Double,
    ) {
        denseRotm.invokeExact(
            MemorySegment.ofArray(x),
            xOff,
            xStride,
            MemorySegment.ofArray(y),
            yOff,
            yStride,
            len,
            h11,
            h12,
            h21,
            h22,
        ) as Unit
    }

    fun sparseDotDense(indices: IntArray, values: DoubleArray, dense: DoubleArray): Double = sparseDotDense.invokeExact(
        MemorySegment.ofArray(indices),
        MemorySegment.ofArray(values),
        indices.size,
        MemorySegment.ofArray(dense),
    ) as Double

    fun sparseDotSparse(aIndices: IntArray, aValues: DoubleArray, bIndices: IntArray, bValues: DoubleArray): Double =
        sparseDotSparse.invokeExact(
            MemorySegment.ofArray(aIndices),
            MemorySegment.ofArray(aValues),
            aIndices.size,
            MemorySegment.ofArray(bIndices),
            MemorySegment.ofArray(bValues),
            bIndices.size,
        ) as Double

    fun sparseAxpy(indices: IntArray, values: DoubleArray, alpha: Double, dense: DoubleArray) {
        sparseAxpy.invokeExact(
            MemorySegment.ofArray(indices),
            MemorySegment.ofArray(values),
            indices.size,
            alpha,
            MemorySegment.ofArray(dense),
        ) as Unit
    }

    fun sparseScatter(indices: IntArray, values: DoubleArray, dense: DoubleArray) {
        sparseScatter.invokeExact(
            MemorySegment.ofArray(indices),
            MemorySegment.ofArray(values),
            indices.size,
            MemorySegment.ofArray(dense),
        ) as Unit
    }

    fun sparseGather(indices: IntArray, values: DoubleArray, dense: DoubleArray) {
        sparseGather.invokeExact(
            MemorySegment.ofArray(indices),
            MemorySegment.ofArray(values),
            indices.size,
            MemorySegment.ofArray(dense),
        ) as Unit
    }

    fun sparseGatherZero(indices: IntArray, values: DoubleArray, dense: DoubleArray) {
        sparseGatherZero.invokeExact(
            MemorySegment.ofArray(indices),
            MemorySegment.ofArray(values),
            indices.size,
            MemorySegment.ofArray(dense),
        ) as Unit
    }

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
