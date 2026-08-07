@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.cblas

import com.eignex.koblas.dense.isIlp64OpenBlas
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

/**
 * The CBLAS subset koblas dispatches to, resolved from the host library with `dlopen`/`dlsym` rather
 * than linked: the OpenBLAS dependency stays a runtime option, so a binary carrying this backend still
 * starts, and falls back to the portable kernels, on hosts without it. Enum parameters are declared
 * `int`, their C ABI, and all integer widths are LP64 — the default OpenBLAS build, not INTERFACE64.
 */
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

/**
 * The LAPACKE subset, resolved separately because it is optional.
 *
 * Debian and Ubuntu strip LAPACKE out of their OpenBLAS build and ship it as liblapacke, a package that
 * libopenblas0 does not pull in. A host with CBLAS and no LAPACKE therefore gets the BLAS half of the
 * backend and the portable LAPACK half, rather than losing native acceleration entirely.
 */
@Suppress("MagicNumber") // the prototypes' arities are ABI facts
internal class LapackeFunctions(private val blas: COpaquePointer, private val lapacke: COpaquePointer?) {
    private fun required(name: String): COpaquePointer = dlsym(blas, name)
        ?: lapacke?.let { dlsym(it, name) }
        ?: error("LAPACKE is present but lacks $name")

    val dgetrf = required("LAPACKE_dgetrf")
        .reinterpret<CFunction<(Int, Int, Int, Dp, Int, Ip) -> Int>>()
    val dgecon = required("LAPACKE_dgecon")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int, Double, Dp) -> Int>>()
    val dgeqrf = required("LAPACKE_dgeqrf")
        .reinterpret<CFunction<(Int, Int, Int, Dp, Int, Dp) -> Int>>()
    val dormqr = required("LAPACKE_dormqr")
        .reinterpret<CFunction<(Int, Byte, Byte, Int, Int, Int, Dp, Int, Dp, Dp, Int) -> Int>>()
    val dpotrf = required("LAPACKE_dpotrf")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int) -> Int>>()
    val dpotrs = required("LAPACKE_dpotrs")
        .reinterpret<CFunction<(Int, Byte, Int, Int, Dp, Int, Dp, Int) -> Int>>()
    val dpotri = required("LAPACKE_dpotri")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int) -> Int>>()
    val dsytrf = required("LAPACKE_dsytrf")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int, Ip) -> Int>>()
    val dsytrs = required("LAPACKE_dsytrs")
        .reinterpret<CFunction<(Int, Byte, Int, Int, Dp, Int, Ip, Dp, Int) -> Int>>()
}

/**
 * Locates the host OpenBLAS once, resolving the two libraries independently.
 *
 * [cblas] is null when there is no usable OpenBLAS at all, in which case koblas stays fully portable.
 * [lapacke] is null when the host has CBLAS but no LAPACKE — the Debian and Ubuntu default, since
 * liblapacke is a separate package. That case keeps the native BLAS half and takes the portable LAPACK
 * half, which is the whole reason the two are resolved apart.
 */
internal object OpenBlasLoader {
    private val handle: COpaquePointer? = open(
        "libopenblas.so.0", // Linux runtime package
        "libopenblas.so",
        "libopenblas.dylib",
        "/opt/homebrew/opt/openblas/lib/libopenblas.dylib", // Homebrew is keg-only
        "/usr/local/opt/openblas/lib/libopenblas.dylib",
    )

    val cblas: CblasFunctions? = handle?.let { blas ->
        // An ILP64 build exports these same names and takes 64-bit integers, so resolution cannot tell it
        // apart from the LP64 one these bindings declare. The config string can.
        if (isIlp64OpenBlas(configString(blas))) return@let null
        val fns = try {
            CblasFunctions(blas)
        } catch (_: IllegalStateException) { // a required symbol is missing: treat as not installed
            return@let null
        }
        // Single-threaded default, the faster configuration at koblas workload sizes; an explicit
        // OPENBLAS_NUM_THREADS is honored by OpenBLAS itself.
        if (getenv("OPENBLAS_NUM_THREADS") == null) fns.setNumThreads?.invoke(1)
        fns
    }

    val lapacke: LapackeFunctions? = handle?.takeIf { cblas != null }?.let { blas ->
        // LAPACKE either lives in the OpenBLAS build or in a separate library; absent both, this half
        // simply is not available.
        val extra = if (dlsym(blas, "LAPACKE_dgetrf") != null) null else open("liblapacke.so.3", "liblapacke.so")
        if (dlsym(blas, "LAPACKE_dgetrf") == null && extra == null) return@let null
        try {
            LapackeFunctions(blas, extra)
        } catch (_: IllegalStateException) {
            null
        }
    }

    /** Whether the host CBLAS resolved — the name every koblas host binding answers availability with. */
    val available: Boolean get() = cblas != null

    /** Whether LAPACKE resolved as well; false on a host that ships CBLAS only. */
    val lapackAvailable: Boolean get() = lapacke != null

    /**
     * The library's `openblas_get_config` string, or empty when it does not offer one.
     *
     * A string read rather than arithmetic, which is what lets it run during resolution where the probe it
     * stands in for could not. A build without the symbol cannot be interrogated, and an empty string reads
     * as "no disqualifying marker".
     */
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
