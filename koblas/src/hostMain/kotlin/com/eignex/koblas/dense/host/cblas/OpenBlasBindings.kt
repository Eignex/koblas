@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.internal.host.openNativeLibrary
import kotlinx.cinterop.*
import platform.posix.dlsym

private typealias Dp = CPointer<DoubleVar>?

/** Every integer here is LP64, matching the default OpenBLAS build rather than an INTERFACE64 one. */
@Suppress("MagicNumber") // the prototypes' arities are ABI facts
internal class CblasFunctions(private val blas: COpaquePointer) {
    private fun required(name: String): COpaquePointer = dlsym(blas, name)
        ?: error("libopenblas is present but lacks $name")

    val setNumThreads = dlsym(blas, "openblas_set_num_threads")
        ?.reinterpret<CFunction<(Int) -> Unit>>()
    val ddot = required("cblas_ddot")
        .reinterpret<CFunction<(Int, Dp, Int, Dp, Int) -> Double>>()
    val dnrm2 = required("cblas_dnrm2")
        .reinterpret<CFunction<(Int, Dp, Int) -> Double>>()
    val dasum = required("cblas_dasum")
        .reinterpret<CFunction<(Int, Dp, Int) -> Double>>()
    val dscal = required("cblas_dscal")
        .reinterpret<CFunction<(Int, Double, Dp, Int) -> Unit>>()
    val dswap = required("cblas_dswap")
        .reinterpret<CFunction<(Int, Dp, Int, Dp, Int) -> Unit>>()
    val daxpy = required("cblas_daxpy")
        .reinterpret<CFunction<(Int, Double, Dp, Int, Dp, Int) -> Unit>>()
    val drotm = required("cblas_drotm")
        .reinterpret<CFunction<(Int, Dp, Int, Dp, Int, Dp) -> Unit>>()
    val drot = required("cblas_drot")
        .reinterpret<CFunction<(Int, Dp, Int, Dp, Int, Double, Double) -> Unit>>()
    val dgemv = required("cblas_dgemv")
        .reinterpret<CFunction<(Int, Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dgemm = required("cblas_dgemm")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dsyrk = required("cblas_dsyrk")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Double, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dsyr2k = required("cblas_dsyr2k")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dsymv = required("cblas_dsymv")
        .reinterpret<CFunction<(Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dsymm = required("cblas_dsymm")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dger = required("cblas_dger")
        .reinterpret<CFunction<(Int, Int, Int, Double, Dp, Int, Dp, Int, Dp, Int) -> Unit>>()
    val dsyr = required("cblas_dsyr")
        .reinterpret<CFunction<(Int, Int, Int, Double, Dp, Int, Dp, Int) -> Unit>>()
    val dsyr2 = required("cblas_dsyr2")
        .reinterpret<CFunction<(Int, Int, Int, Double, Dp, Int, Dp, Int, Dp, Int) -> Unit>>()
    val dtrmv = required("cblas_dtrmv")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Dp, Int, Dp, Int) -> Unit>>()
    val dtrmm = required("cblas_dtrmm")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Int, Int, Double, Dp, Int, Dp, Int) -> Unit>>()
    val dtrsv = required("cblas_dtrsv")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Dp, Int, Dp, Int) -> Unit>>()
    val dtrsm = required("cblas_dtrsm")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Int, Int, Double, Dp, Int, Dp, Int) -> Unit>>()
}

internal class OpenBlasLoader(private val config: HostBlasConfig = HostBlasConfig()) {
    companion object {
        private val defaultLoader: OpenBlasLoader by lazy { OpenBlasLoader() }

        val cblas: CblasFunctions? get() = defaultLoader.cblas
    }
    private val handle: COpaquePointer? = openNativeLibrary(config.libraryPath?.let(::listOf) ?: OPENBLAS_SONAMES)

    private var configuredThreadCount: Int? = null

    /** Thread count applied through OpenBLAS, or null when it was not requested or the setter is absent. */
    val effectiveThreadCount: Int? get() = configuredThreadCount

    val cblas: CblasFunctions? = handle?.let { blas ->
        // An ILP64 build exports these same names but takes 64-bit integers. Require OpenBLAS's configuration
        // string to identify the library and its ABI before binding the LP64 prototypes below.
        if (!isLp64OpenBlas(configString(blas))) return@let null
        val fns = try {
            CblasFunctions(blas)
        } catch (_: IllegalStateException) { // a required symbol is missing, treat as not installed
            return@let null
        }
        config.threadCount?.let { count ->
            fns.setNumThreads?.let { setter ->
                setter(count)
                configuredThreadCount = count
            }
        }
        fns
    }

    /** The library's openblas_get_config string, or empty when the build does not offer one. */
    private fun configString(blas: COpaquePointer): String {
        val symbol = dlsym(blas, "openblas_get_config") ?: return ""
        val text = symbol.reinterpret<CFunction<() -> CPointer<ByteVar>?>>().invoke()
        return text?.toKString() ?: ""
    }
}
