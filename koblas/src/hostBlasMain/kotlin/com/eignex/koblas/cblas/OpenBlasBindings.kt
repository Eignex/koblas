@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import kotlinx.cinterop.CFunction
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.invoke
import kotlinx.cinterop.reinterpret
import platform.posix.RTLD_NOW
import platform.posix.dlopen
import platform.posix.dlsym
import platform.posix.getenv

private typealias Dp = CPointer<DoubleVar>?
private typealias Ip = CPointer<IntVar>?

/**
 * The double-precision CBLAS/LAPACKE subset koblas dispatches to, resolved from the host libraries
 * with `dlopen`/`dlsym` at first use rather than linked: the OpenBLAS dependency stays a runtime
 * option, so a binary that carries this backend still starts (and falls back to the reference
 * implementation) on hosts without it. Enum parameters are declared `int`, their C ABI, and all
 * integer widths are LP64 (32-bit `lapack_int`) — the default OpenBLAS build, not INTERFACE64.
 */
@Suppress("MagicNumber") // the prototypes' arities are ABI facts
internal class OpenBlasFunctions(private val blas: COpaquePointer, private val lapacke: COpaquePointer?) {
    private fun required(name: String): COpaquePointer = dlsym(blas, name)
        ?: lapacke?.let { dlsym(it, name) }
        ?: error("libopenblas is present but lacks $name")

    /** Absent from some builds; threading setup is best-effort. */
    val setNumThreads =
        dlsym(blas, "openblas_set_num_threads")?.reinterpret<CFunction<(Int) -> Unit>>()

    val ddot = required("cblas_ddot")
        .reinterpret<CFunction<(Int, Dp, Int, Dp, Int) -> Double>>()
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
    val dtrsv = required("cblas_dtrsv")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Dp, Int, Dp, Int) -> Unit>>()
    val dtrsm = required("cblas_dtrsm")
        .reinterpret<CFunction<(Int, Int, Int, Int, Int, Int, Int, Double, Dp, Int, Dp, Int) -> Unit>>()
    val dgetrf = required("LAPACKE_dgetrf")
        .reinterpret<CFunction<(Int, Int, Int, Dp, Int, Ip) -> Int>>()
    val dgecon = required("LAPACKE_dgecon")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int, Double, Dp) -> Int>>()
    val dgeqrf = required("LAPACKE_dgeqrf")
        .reinterpret<CFunction<(Int, Int, Int, Dp, Int, Dp) -> Int>>()
    val dormqr = required("LAPACKE_dormqr")
        .reinterpret<CFunction<(Int, Byte, Byte, Int, Int, Int, Dp, Int, Dp, Dp, Int) -> Int>>()
    val dsytrf = required("LAPACKE_dsytrf")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int, Ip) -> Int>>()
    val dsytrs = required("LAPACKE_dsytrs")
        .reinterpret<CFunction<(Int, Byte, Int, Int, Dp, Int, Ip, Dp, Int) -> Int>>()
}

/** Locates the host OpenBLAS once; [functions] is null when the host has no usable installation. */
internal object OpenBlasLoader {
    val functions: OpenBlasFunctions? = load()

    private fun load(): OpenBlasFunctions? {
        val blas = open(
            "libopenblas.so.0", // Linux runtime package
            "libopenblas.so",
            "libopenblas.dylib",
            "/opt/homebrew/opt/openblas/lib/libopenblas.dylib", // Homebrew is keg-only
            "/usr/local/opt/openblas/lib/libopenblas.dylib",
        ) ?: return null
        // Debian/Ubuntu strip LAPACKE out of their OpenBLAS build; it ships as liblapacke, which
        // dispatches to the distribution's active LAPACK alternative.
        val lapacke = if (dlsym(blas, "LAPACKE_dgetrf") != null) {
            null
        } else {
            open("liblapacke.so.3", "liblapacke.so") ?: return null
        }
        val fns = try {
            OpenBlasFunctions(blas, lapacke)
        } catch (_: IllegalStateException) { // a required symbol is missing: treat as not installed
            return null
        }
        // Single-threaded default, the faster configuration at koblas workload sizes; an explicit
        // OPENBLAS_NUM_THREADS is honored by OpenBLAS itself.
        if (getenv("OPENBLAS_NUM_THREADS") == null) fns.setNumThreads?.invoke(1)
        return fns
    }

    private fun open(vararg names: String): COpaquePointer? {
        for (name in names) dlopen(name, RTLD_NOW)?.let { return it }
        return null
    }
}
