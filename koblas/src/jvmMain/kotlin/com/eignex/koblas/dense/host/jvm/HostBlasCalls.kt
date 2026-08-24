package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.cblas.LAPACKE_SONAMES
import com.eignex.koblas.dense.host.cblas.OPENBLAS_SONAMES
import com.eignex.koblas.dense.host.cblas.isIlp64OpenBlas
import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.internal.host.FfmLibrary.Companion.doubleOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.intOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.pointerOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.voidOf
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/**
 * The LP64 CBLAS and LAPACKE subset, bound as downcalls and invoked with `invokeExact`, so an argument's
 * Kotlin type has to match its layout exactly. See [com.eignex.koblas.internal.host.FfmLibrary].
 */
internal class HostBlasCalls(internal val config: HostBlasConfig) {

    /** Whether the host's CBLAS resolved and takes the integer width koblas binds. */
    val available: Boolean

    /** Whether LAPACKE resolved as well; false on a host that ships CBLAS only. */
    val lapackAvailable: Boolean

    private val openblasNames = config.libraryPath?.let(::listOf) ?: OPENBLAS_SONAMES

    private val lapackeNames = config.lapackeLibraryPath?.let(::listOf) ?: LAPACKE_SONAMES

    /** The OpenBLAS build, and behind it the separate liblapacke a host that needs one keeps LAPACKE in. */
    private val library: FfmLibrary

    /**
     * Every CBLAS entry point these bindings resolve. Availability is the whole set rather than one symbol,
     * because the handles bind lazily and a missing one raises past the dispatch gate, where nothing is left
     * to fall back to. A host offering part of the library has to leave the half portable instead. Presence
     * is read with `find`, which is a lookup; binding would be the stack-hungry thing discovery avoids.
     */
    private val requiredCblas = listOf(
        "cblas_dasum", "cblas_daxpy", "cblas_ddot", "cblas_dgemm", "cblas_dgemv", "cblas_dger",
        "cblas_dnrm2", "cblas_dscal", "cblas_dsymm", "cblas_dsymv", "cblas_dsyrk", "cblas_dtrmm",
        "cblas_dtrmv", "cblas_dtrsm", "cblas_dtrsv",
    )

    /** The same for LAPACKE, less `LAPACKE_dgeqp3`, which [optionalHandle] already lets a host omit. */
    private val requiredLapacke = listOf(
        "LAPACKE_dgecon",
        "LAPACKE_dgeqrf",
        "LAPACKE_dgetrf",
        "LAPACKE_dormqr",
        "LAPACKE_dpotrf",
        "LAPACKE_dpotri",
        "LAPACKE_dsytrf",
        "LAPACKE_dsytrs",
    )

    init {
        val blas = FfmLibrary.open(openblasNames, KEY_CBLAS_SYMBOL, "the host OpenBLAS")
        val resolved = blas.containsAll(requiredCblas)
        // An ILP64 build exports these same names, so only the config string tells it apart from the LP64
        // one these bindings declare.
        val ilp64 = resolved && isIlp64OpenBlas(configString(blas))
        available = resolved && !ilp64
        // A second library for the hosts that ship LAPACKE outside their OpenBLAS build.
        val lapackeInBlas = available && blas.contains(KEY_LAPACKE_SYMBOL)
        val extra = if (available && !lapackeInBlas) {
            FfmLibrary.open(lapackeNames, KEY_LAPACKE_SYMBOL, "the host LAPACKE")
        } else {
            null
        }
        // The OpenBLAS build first and then liblapacke, so a host splitting the set across the two is served.
        library = blas.withFallback(extra)
        lapackAvailable = available && library.containsAll(requiredLapacke)
        if (available) configureThreads()
    }

    /** OpenBLAS owns this setting process-wide; setting it at backend construction makes that scope explicit. */
    private fun configureThreads() {
        val count = config.threadCount ?: return
        // Not a critical downcall: setting the count starts OpenBLAS's threads, which one may not do.
        val setter = library.handleOrNull("openblas_set_num_threads", voidOf(JAVA_INT), critical = false) ?: return
        setter.invokeExact(count) as Unit
    }

    /** The library's openblas_get_config string, or empty when it does not offer one. */
    private fun configString(blas: FfmLibrary): String {
        val handle = blas.handleOrNull("openblas_get_config", pointerOf(), critical = false) ?: return ""
        val text = handle.invokeExact() as MemorySegment
        if (text.address() == 0L) return ""
        // The returned pointer carries no length, so it is re-sized before the string is read.
        return text.reinterpret(Long.MAX_VALUE).getString(0)
    }

    private fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle = library.handle(name, descriptor)

    /** Looked up the same way as [handle] but tolerating absence, for a routine koblas can do without. */
    private fun optionalHandle(name: String, descriptor: FunctionDescriptor): MethodHandle? =
        library.handleOrNull(name, descriptor)

    // Enum arguments are int, their C ABI, and lapack_int is 32-bit, matching the default (non-INTERFACE64)
    // OpenBLAS build.
    val dgemv: MethodHandle by lazy {
        handle(
            "cblas_dgemv",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
                ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }

    val dsymv: MethodHandle by lazy {
        handle(
            "cblas_dsymv",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
                ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }

    val dger: MethodHandle by lazy {
        handle(
            "cblas_dger",
            voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }

    val dtrsv: MethodHandle by lazy {
        handle(
            "cblas_dtrsv",
            voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }

    val dtrmv: MethodHandle by lazy {
        handle(
            "cblas_dtrmv",
            voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
        )
    }

    val dtrmm: MethodHandle by lazy {
        handle(
            "cblas_dtrmm",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
            ),
        )
    }

    val dpotrf: MethodHandle by lazy {
        handle("LAPACKE_dpotrf", intOf(JAVA_INT, JAVA_BYTE, JAVA_INT, ADDRESS, JAVA_INT))
    }

    val dpotri: MethodHandle by lazy {
        handle("LAPACKE_dpotri", intOf(JAVA_INT, JAVA_BYTE, JAVA_INT, ADDRESS, JAVA_INT))
    }

    val dgemm: MethodHandle by lazy {
        handle(
            "cblas_dgemm",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }

    val dsyrk: MethodHandle by lazy {
        handle(
            "cblas_dsyrk",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }

    val dsymm: MethodHandle by lazy {
        handle(
            "cblas_dsymm",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        )
    }

    val dtrsm: MethodHandle by lazy {
        handle(
            "cblas_dtrsm",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
            ),
        )
    }

    val dscal: MethodHandle by lazy { handle("cblas_dscal", voidOf(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT)) }

    val daxpy: MethodHandle by lazy {
        handle("cblas_daxpy", voidOf(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
    }

    val ddot: MethodHandle by lazy {
        handle("cblas_ddot", doubleOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
    }

    val dnrm2: MethodHandle by lazy { handle("cblas_dnrm2", doubleOf(JAVA_INT, ADDRESS, JAVA_INT)) }

    val dasum: MethodHandle by lazy { handle("cblas_dasum", doubleOf(JAVA_INT, ADDRESS, JAVA_INT)) }

    val dgetrf: MethodHandle by lazy {
        handle("LAPACKE_dgetrf", intOf(JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS))
    }

    val dgecon: MethodHandle by lazy {
        handle("LAPACKE_dgecon", intOf(JAVA_INT, JAVA_BYTE, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS))
    }

    val dgeqrf: MethodHandle by lazy {
        handle("LAPACKE_dgeqrf", intOf(JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS))
    }

    /**
     * int LAPACKE_dgeqp3(int layout, int m, int n, double* a, int lda, int* jpvt, double* tau)
     *
     * Optional, so a LAPACKE without it keeps the rest of the half rather than failing every call that
     * reaches it. The pivoted QR is the only routine that needs it and koblas has a portable one, which is
     * how the native binding treats it too.
     */
    val dgeqp3: MethodHandle? by lazy {
        optionalHandle("LAPACKE_dgeqp3", intOf(JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS))
    }

    val dormqr: MethodHandle by lazy {
        handle(
            "LAPACKE_dormqr",
            intOf(
                JAVA_INT, JAVA_BYTE, JAVA_BYTE, JAVA_INT, JAVA_INT, JAVA_INT,
                ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT,
            ),
        )
    }

    val dsytrf: MethodHandle by lazy {
        handle("LAPACKE_dsytrf", intOf(JAVA_INT, JAVA_BYTE, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS))
    }

    val dsytrs: MethodHandle by lazy {
        handle(
            "LAPACKE_dsytrs",
            intOf(JAVA_INT, JAVA_BYTE, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_INT),
        )
    }

    private companion object {
        /** Read to tell an OpenBLAS build apart from a library that merely carries the soname. */
        const val KEY_CBLAS_SYMBOL = "cblas_dgemm"

        /** Read to find which of the two libraries LAPACKE lives in. */
        const val KEY_LAPACKE_SYMBOL = "LAPACKE_dgetrf"
    }

    /** Pins [values] for the call. Nothing is copied, and the address is valid only for that call. */
    fun seg(values: DoubleArray): MemorySegment = MemorySegment.ofArray(values)

    /** Pins [values] for the call. */
    fun seg(values: IntArray): MemorySegment = MemorySegment.ofArray(values)
}
