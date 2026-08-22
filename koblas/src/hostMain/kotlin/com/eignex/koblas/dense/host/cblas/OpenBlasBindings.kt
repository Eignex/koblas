@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.dense.host.cblas.isIlp64OpenBlas
import com.eignex.koblas.internal.backend.ConfigurationKeys
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CFunction
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.invoke
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.toKString
import platform.posix.RTLD_NOW
import platform.posix.dlopen
import platform.posix.dlsym
import platform.posix.getenv

private typealias Dp = CPointer<DoubleVar>?
private typealias Ip = CPointer<IntVar>?

/** Every integer here is LP64, matching the default OpenBLAS build rather than an INTERFACE64 one. */
@Suppress("MagicNumber") // the prototypes' arities are ABI facts
internal class CblasFunctions(private val blas: COpaquePointer) {
    private fun required(name: String): COpaquePointer = dlsym(blas, name)
        ?: error("libopenblas is present but lacks $name")

    /** Absent from some builds; threading setup is best-effort. */
    val setNumThreads =
        dlsym(blas, "openblas_set_num_threads")?.reinterpret<CFunction<(Int) -> Unit>>()
    val ddot = required("cblas_ddot")
        .reinterpret<CFunction<(Int, Dp, Int, Dp, Int) -> Double>>()
    val dnrm2 = required("cblas_dnrm2")
        .reinterpret<CFunction<(Int, Dp, Int) -> Double>>()
    val dasum = required("cblas_dasum")
        .reinterpret<CFunction<(Int, Dp, Int) -> Double>>()
    val dscal = required("cblas_dscal")
        .reinterpret<CFunction<(Int, Double, Dp, Int) -> Unit>>()
    val daxpy = required("cblas_daxpy")
        .reinterpret<CFunction<(Int, Double, Dp, Int, Dp, Int) -> Unit>>()
    val dgemv = required("cblas_dgemv")
        .reinterpret<CFunction<(Int, Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dgemm = required("cblas_dgemm")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dsyrk = required("cblas_dsyrk")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Double, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dsymv = required("cblas_dsymv")
        .reinterpret<CFunction<(Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dsymm = required("cblas_dsymm")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Double, Dp, Int, Dp, Int, Double, Dp, Int) -> Unit>>()
    val dger = required("cblas_dger")
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

/** Resolved separately from CBLAS, so a host with CBLAS and no LAPACKE keeps the portable LAPACK half. */
@Suppress("MagicNumber") // the prototypes' arities are ABI facts
internal class LapackeFunctions(private val blas: COpaquePointer, private val lapacke: COpaquePointer?) {
    private fun required(name: String): COpaquePointer = dlsym(blas, name)
        ?: lapacke?.let { dlsym(it, name) }
        ?: error("LAPACKE is present but lacks $name")

    /** Looked up the same way as [required] but tolerating absence, for a routine koblas can do without. */
    private fun optional(name: String): COpaquePointer? = dlsym(blas, name) ?: lapacke?.let { dlsym(it, name) }

    val dgetrf = required("LAPACKE_dgetrf")
        .reinterpret<CFunction<(Int, Int, Int, Dp, Int, Ip) -> Int>>()
    val dgecon = required("LAPACKE_dgecon")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int, Double, Dp) -> Int>>()
    val dgeqrf = required("LAPACKE_dgeqrf")
        .reinterpret<CFunction<(Int, Int, Int, Dp, Int, Dp) -> Int>>()

    /**
     * Optional, so a LAPACKE without it keeps the rest of the half rather than disabling all of it. The
     * pivoted QR is the only routine that needs it, and koblas has a portable one.
     */
    val dgeqp3 = optional("LAPACKE_dgeqp3")
        ?.reinterpret<CFunction<(Int, Int, Int, Dp, Int, Ip, Dp) -> Int>>()
    val dormqr = required("LAPACKE_dormqr")
        .reinterpret<CFunction<(Int, Byte, Byte, Int, Int, Int, Dp, Int, Dp, Dp, Int) -> Int>>()
    val dpotrf = required("LAPACKE_dpotrf")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int) -> Int>>()
    val dpotri = required("LAPACKE_dpotri")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int) -> Int>>()
    val dsytrf = required("LAPACKE_dsytrf")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int, Ip) -> Int>>()
    val dsytrs = required("LAPACKE_dsytrs")
        .reinterpret<CFunction<(Int, Byte, Int, Int, Dp, Int, Ip, Dp, Int) -> Int>>()
}

internal object OpenBlasLoader {
    private val handle: COpaquePointer? = open(
        "libopenblas.so.0", // Linux runtime package
        "libopenblas.so",
        "libopenblas.dylib",
        "/opt/homebrew/opt/openblas/lib/libopenblas.dylib", // Homebrew is keg-only
        "/usr/local/opt/openblas/lib/libopenblas.dylib",
    )

    val cblas: CblasFunctions? = handle?.let { blas ->
        // An ILP64 build exports these same names but takes 64-bit integers; only the config string tells.
        if (isIlp64OpenBlas(configString(blas))) return@let null
        val fns = try {
            CblasFunctions(blas)
        } catch (_: IllegalStateException) { // a required symbol is missing, treat as not installed
            return@let null
        }
        // Single-threaded is the faster configuration at koblas workload sizes.
        if (getenv(ConfigurationKeys.OPENBLAS_THREADS_ENV) == null) fns.setNumThreads?.invoke(1)
        fns
    }

    val lapacke: LapackeFunctions? = handle?.takeIf { cblas != null }?.let { blas ->
        // LAPACKE lives either in the OpenBLAS build or in a separate library.
        val extra = if (dlsym(blas, "LAPACKE_dgetrf") != null) null else open("liblapacke.so.3", "liblapacke.so")
        if (dlsym(blas, "LAPACKE_dgetrf") == null && extra == null) return@let null
        try {
            LapackeFunctions(blas, extra)
        } catch (_: IllegalStateException) {
            null
        }
    }

    /** The library's openblas_get_config string, or empty when the build does not offer one. */
    private fun configString(blas: COpaquePointer): String {
        val symbol = dlsym(blas, "openblas_get_config") ?: return ""
        val text = symbol.reinterpret<CFunction<() -> CPointer<ByteVar>?>>().invoke()
        return text?.toKString() ?: ""
    }

    private fun open(vararg names: String): COpaquePointer? {
        for (name in names) dlopen(name, RTLD_NOW)?.let { return it }
        return null
    }
}
