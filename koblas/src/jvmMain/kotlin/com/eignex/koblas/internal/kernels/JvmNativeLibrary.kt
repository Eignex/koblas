package com.eignex.koblas.internal.kernels

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions

/** Owns the loaded library; capability inspection never uses a critical downcall. */
internal object JvmNativeLibrary {
    private val symbolNames = listOf(
        "koblas_probe_v1",
        "koblas_dense_dot_v1",
        "koblas_dense_ssqd_v1",
        "koblas_dense_axpy_v1",
        "koblas_dense_axpy_arithmetic_v1",
        "koblas_dense_scale_v1",
        "koblas_dense_nrm2_v1",
        "koblas_dense_sum_v1",
        "koblas_dense_asum_v1",
        "koblas_dense_iamax_v1",
        "koblas_dense_gemm_tile_v1",
        "koblas_dense_swap_v1",
        "koblas_dense_dot4_v1",
        "koblas_dense_axpy4_v1",
        "koblas_dense_dot_axpy_v1",
        "koblas_dense_rotm_v1",
        "koblas_sparse_dot_dense_v1",
        "koblas_sparse_dot_sparse_v1",
        "koblas_sparse_axpy_v1",
        "koblas_sparse_scatter_v1",
        "koblas_sparse_nrm2_v1",
        "koblas_sparse_gather_v1",
        "koblas_sparse_gather_zero_v1",
        "koblas_dense_trsm_tile_v1",
        "koblas_dense_gemm_trsm_tile_v1",
    )
    val library: FfmLibrary? = loadLibraryOrNull()

    private fun loadLibraryOrNull(): FfmLibrary? = try {
        val extractedLibrary = extractLibrary()
        FfmLibrary.open(
            listOf(extractedLibrary.toString()),
            "koblas_probe_v1",
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
            ?: JvmNativeLibrary::class.java.classLoader.getResourceAsStream(resource)
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
