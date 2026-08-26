@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.internal.host.openNativeLibrary
import kotlinx.cinterop.*
import platform.posix.dlsym

private typealias Dp = CPointer<DoubleVar>?
private typealias Ip = CPointer<IntVar>?

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
    val dgetri = required("LAPACKE_dgetri")
        .reinterpret<CFunction<(Int, Int, Dp, Int, Ip) -> Int>>()
    val dtrtri = required("LAPACKE_dtrtri")
        .reinterpret<CFunction<(Int, Byte, Byte, Int, Dp, Int) -> Int>>()
    val dpotrf = required("LAPACKE_dpotrf")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int) -> Int>>()
    val dpotri = required("LAPACKE_dpotri")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int) -> Int>>()
    val dsytrf = required("LAPACKE_dsytrf")
        .reinterpret<CFunction<(Int, Byte, Int, Dp, Int, Ip) -> Int>>()
    val dsytrs = required("LAPACKE_dsytrs")
        .reinterpret<CFunction<(Int, Byte, Int, Int, Dp, Int, Ip, Dp, Int) -> Int>>()
}

internal class OpenBlasLoader(private val config: HostBlasConfig = HostBlasConfig()) {
    companion object {
        private val defaultLoader: OpenBlasLoader by lazy { OpenBlasLoader() }

        val cblas: CblasFunctions? get() = defaultLoader.cblas
        val lapacke: LapackeFunctions? get() = defaultLoader.lapacke
    }
    private val handle: COpaquePointer? = openNativeLibrary(config.libraryPath?.let(::listOf) ?: OPENBLAS_SONAMES)

    val cblas: CblasFunctions? = handle?.let { blas ->
        // An ILP64 build exports these same names but takes 64-bit integers, so the config string and then
        // the width of the pivots it writes are what tell it apart from the LP64 one these bindings declare.
        if (isIlp64Build(configString(blas)) { probePivots(blas) }) return@let null
        val fns = try {
            CblasFunctions(blas)
        } catch (_: IllegalStateException) { // a required symbol is missing, treat as not installed
            return@let null
        }
        config.threadCount?.let { fns.setNumThreads?.invoke(it) }
        fns
    }

    val lapacke: LapackeFunctions? = handle?.takeIf { cblas != null }?.let { blas ->
        // LAPACKE lives either in the OpenBLAS build or in a separate library.
        val extra = if (dlsym(blas, "LAPACKE_dgetrf") != null) {
            null
        } else {
            openNativeLibrary(config.lapackeLibraryPath?.let(::listOf) ?: LAPACKE_SONAMES)
        }
        if (dlsym(blas, "LAPACKE_dgetrf") == null && extra == null) return@let null
        // A separate liblapacke of the wrong width costs the host only its LAPACK half, since the CBLAS it
        // sits behind was judged on its own.
        if (extra != null && isIlp64PivotWidth(probePivots(extra) ?: IntArray(0))) return@let null
        try {
            LapackeFunctions(blas, extra)
        } catch (_: IllegalStateException) {
            null
        }
    }

    /**
     * The pivot words [blas] writes for a probing `LAPACKE_dgetrf`, or null when it carries no such routine
     * or the factorization failed. Every integer argument is passed as a long whose upper half is zero,
     * which both widths read correctly, so the only thing the call turns on is how wide the pivots it writes
     * are. The buffer holds two 64-bit pivots, so an ILP64 build has room for what it writes.
     */
    private fun probePivots(blas: COpaquePointer): IntArray? {
        val symbol = dlsym(blas, "LAPACKE_dgetrf") ?: return null
        val getrf = symbol.reinterpret<CFunction<(Long, Long, Long, Dp, Long, CPointer<LongVar>?) -> Int>>()
        memScoped {
            // Column-major [[0, 1], [1, 0]], so partial pivoting selects row 2 at both steps.
            val a = allocArray<DoubleVar>(PROBE_ORDER * PROBE_ORDER)
            a[0] = 0.0
            a[1] = 1.0
            a[2] = 1.0
            a[3] = 0.0
            val pivots = allocArray<LongVar>(PROBE_WORDS)
            for (k in 0 until PROBE_WORDS) pivots[k] = 0L
            val order = PROBE_ORDER.toLong()
            val status = getrf(Cblas.COL_MAJOR.toLong(), order, order, a, order, pivots)
            if (status != 0) return null
            val words = pivots.reinterpret<IntVar>()
            return IntArray(PROBE_WORDS) { words[it] }
        }
    }

    /** The library's openblas_get_config string, or empty when the build does not offer one. */
    private fun configString(blas: COpaquePointer): String {
        val symbol = dlsym(blas, "openblas_get_config") ?: return ""
        val text = symbol.reinterpret<CFunction<() -> CPointer<ByteVar>?>>().invoke()
        return text?.toKString() ?: ""
    }
}
