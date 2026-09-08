package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.cblas.*
import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.internal.host.FfmLibrary.Companion.doubleOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.pointerOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.voidOf
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

/**
 * The LP64 CBLAS subset, bound as downcalls and invoked with `invokeExact`, so an argument's
 * Kotlin type has to match its layout exactly. See [com.eignex.koblas.internal.host.FfmLibrary].
 */
internal class HostBlasCalls(internal val config: HostBlasConfig) {

    /** Whether the host's CBLAS resolved and takes the integer width koblas binds. */
    val available: Boolean

    /** Thread count applied through OpenBLAS, or null when it was not requested or the setter is absent. */
    val effectiveThreadCount: Int?

    private val openblasNames = config.libraryPath?.let(::listOf) ?: OPENBLAS_SONAMES

    /** The OpenBLAS build. */
    private val library: FfmLibrary

    /**
     * Every CBLAS entry point these bindings resolve. Availability is the whole set rather than one symbol,
     * because the handles bind lazily and a missing one raises on the first call, where nothing is left
     * to fall back to. A host offering part of the library has to leave the half portable instead. Presence
     * is read with `find`, which is a lookup; binding would be the stack-hungry thing discovery avoids.
     */
    private val requiredCblas = listOf(
        "cblas_dasum", "cblas_daxpy", "cblas_ddot", "cblas_dgemm", "cblas_dgemv", "cblas_dger", "cblas_drot",
        "cblas_drotm",
        "cblas_dswap", "cblas_dnrm2", "cblas_dscal", "cblas_dsymm", "cblas_dsymv", "cblas_dsyr", "cblas_dsyr2",
        "cblas_dsyr2k", "cblas_dsyrk", "cblas_dtrmm",
        "cblas_dtrmv", "cblas_dtrsm", "cblas_dtrsv",
    )

    init {
        val blas = FfmLibrary.open(openblasNames, KEY_CBLAS_SYMBOL, "the host OpenBLAS")
        val resolved = blas.containsAll(requiredCblas)
        // An ILP64 build exports these same names. Require OpenBLAS's configuration string to identify the
        // library and its ABI before binding the LP64 prototypes below.
        val blasLp64 = resolved && isLp64OpenBlas(configString(blas))
        available = resolved && blasLp64
        library = blas
        effectiveThreadCount = if (available) configureThreads() else null
    }

    /** OpenBLAS owns this setting process-wide; setting it at backend construction makes that scope explicit. */
    private fun configureThreads(): Int? {
        val count = config.threadCount ?: return null
        // Not a critical downcall: setting the count starts OpenBLAS's threads, which one may not do.
        val setter = library.handleOrNull("openblas_set_num_threads", voidOf(JAVA_INT), critical = false) ?: return null
        setter.invokeExact(count) as Unit
        return count
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

    // Enum and dimension arguments are int, matching the default (non-INTERFACE64) OpenBLAS build.
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

    val dsyr: MethodHandle by lazy {
        handle("cblas_dsyr", voidOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
    }

    val dsyr2: MethodHandle by lazy {
        handle(
            "cblas_dsyr2",
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

    val dsyr2k: MethodHandle by lazy {
        handle(
            "cblas_dsyr2k",
            voidOf(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
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

    val dswap: MethodHandle by lazy {
        handle("cblas_dswap", voidOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
    }

    val daxpy: MethodHandle by lazy {
        handle("cblas_daxpy", voidOf(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
    }

    val ddot: MethodHandle by lazy {
        handle("cblas_ddot", doubleOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
    }

    val dnrm2: MethodHandle by lazy { handle("cblas_dnrm2", doubleOf(JAVA_INT, ADDRESS, JAVA_INT)) }

    val dasum: MethodHandle by lazy { handle("cblas_dasum", doubleOf(JAVA_INT, ADDRESS, JAVA_INT)) }

    val drotm: MethodHandle by lazy {
        handle("cblas_drotm", voidOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS))
    }

    val drot: MethodHandle by lazy {
        handle("cblas_drot", voidOf(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE))
    }

    private companion object {
        /** Read to tell an OpenBLAS build apart from a library that merely carries the soname. */
        const val KEY_CBLAS_SYMBOL = "cblas_dgemm"
    }
}
