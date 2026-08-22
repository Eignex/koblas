package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.ConfigurationKeys
import com.eignex.koblas.dense.isIlp64OpenBlas
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemoryLayout
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/**
 * The LP64 CBLAS and LAPACKE subset, bound as downcalls that are invoked with `invokeWithArguments`, since
 * Kotlin cannot emit signature-polymorphic `invokeExact` and the native side would then read garbage.
 */
internal object HostBlasCalls {

    /** Whether the host's CBLAS resolved and takes the integer width koblas binds. */
    val available: Boolean

    /** Whether LAPACKE resolved as well; false on a host that ships CBLAS only. */
    val lapackAvailable: Boolean

    /**
     * Null where this platform has no native linker to hand out. Resolved defensively because this object's
     * initializer runs on the first backend discovery, and an escape from it would leave the portable
     * reference path, which calls out to nothing, unreachable behind a failed class initialization.
     */
    private val nativeLinker: Linker? = try {
        Linker.nativeLinker()
    } catch (_: UnsupportedOperationException) {
        null
    }

    private val linker: Linker get() = requireNotNull(nativeLinker)

    // Pins the on-heap arrays handed over as segments for the call instead of copying them, blocking
    // relocation while it runs.
    private val critical = Linker.Option.critical(true)

    private var lookup: SymbolLookup? = null

    /** Where LAPACKE lives when the OpenBLAS build does not include it. */
    private var lapackeLookup: SymbolLookup? = null

    private val SONAMES = listOf(
        "libopenblas.so.0",
        "libopenblas.so",
        "libopenblas.dylib",
        "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
        "/usr/local/opt/openblas/lib/libopenblas.dylib",
        "openblas.dll",
    )

    private val LAPACKE_SONAMES = listOf("liblapacke.so.3", "liblapacke.so", "liblapacke.dylib", "lapacke.dll")

    /**
     * Every CBLAS entry point these bindings resolve. Availability is the whole set rather than one symbol,
     * because the handles bind lazily and a missing one raises past the dispatch gate, where nothing is left
     * to fall back to. A host offering part of the library has to leave the half portable instead. Presence
     * is read with `find`, which is a lookup; binding would be the stack-hungry thing discovery avoids.
     */
    private val REQUIRED_CBLAS = listOf(
        "cblas_daxpy", "cblas_dgemm", "cblas_dgemv", "cblas_dger", "cblas_dscal", "cblas_dsymm",
        "cblas_dsymv", "cblas_dsyrk", "cblas_dtrmm", "cblas_dtrmv", "cblas_dtrsm", "cblas_dtrsv",
    )

    /** The same for LAPACKE, less `LAPACKE_dgeqp3`, which [optionalHandle] already lets a host omit. */
    private val REQUIRED_LAPACKE = listOf(
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
        val blas = if (nativeLinker == null) null else openLibrary(SONAMES)
        lookup = blas
        val resolved = blas != null && REQUIRED_CBLAS.all { blas.find(it).isPresent }
        // An ILP64 build exports these same names, so only the config string tells it apart from the LP64
        // one these bindings declare.
        val ilp64 = resolved && isIlp64OpenBlas(configString(requireNotNull(blas)))
        available = resolved && !ilp64
        val lapackeInBlas = available && requireNotNull(blas).find("LAPACKE_dgetrf").isPresent
        // A second library for the hosts that ship LAPACKE outside their OpenBLAS build.
        val extra = if (available && !lapackeInBlas) openLibrary(LAPACKE_SONAMES) else null
        lapackeLookup = extra
        // Resolved the way [symbol] resolves, the OpenBLAS build first and then liblapacke, so a host
        // splitting the set across the two is still served.
        lapackAvailable = available && REQUIRED_LAPACKE.all { name ->
            blas?.find(name)?.isPresent == true || extra?.find(name)?.isPresent == true
        }
        // Before any routine runs: an unconfigured OpenBLAS is multithreaded and its parallel LAPACK
        // overflows a default JVM thread stack, killing the process with an uncatchable SIGSEGV.
        if (available) configureThreads()
    }

    /** The library's openblas_get_config string, or empty when it does not offer one. */
    private fun configString(blas: SymbolLookup): String {
        val symbol = blas.find("openblas_get_config").orElse(null) ?: return ""
        val handle = linker.downcallHandle(symbol, FunctionDescriptor.of(ADDRESS))
        val text = handle.invokeWithArguments() as MemorySegment
        if (text.address() == 0L) return ""
        // The returned pointer carries no length, so it is re-sized before the string is read.
        return text.reinterpret(Long.MAX_VALUE).getString(0)
    }

    /**
     * Pins OpenBLAS to one thread unless `koblas.openblas.threads` asks for a count or an explicit
     * OPENBLAS_NUM_THREADS is set, which the library honors itself.
     */
    private fun configureThreads() {
        val requested = System.getProperty(ConfigurationKeys.OPENBLAS_THREADS_PROPERTY)?.toIntOrNull()
        when {
            requested != null -> setThreads(requested)
            System.getenv(ConfigurationKeys.OPENBLAS_THREADS_ENV) == null -> setThreads(1)
        }
    }

    /** Opens the first of [names] the platform loader can find, or null when none resolves. */
    private fun openLibrary(names: List<String>): SymbolLookup? = try {
        names.firstNotNullOfOrNull { runCatching { SymbolLookup.libraryLookup(it, Arena.global()) }.getOrNull() }
    } catch (_: Throwable) { // a host without the library must not crash startup
        null
    }

    /**
     * Configures threads on this instance, which need not be the handle any other preset configured. An
     * unconfigured OpenBLAS runs multithreaded and its parallel LAPACK overflows JVM thread stacks.
     */
    private fun setThreads(count: Int) {
        val setter = requireNotNull(lookup).find("openblas_set_num_threads").orElse(null) ?: return
        linker.downcallHandle(setter, FunctionDescriptor.ofVoid(JAVA_INT)).invokeWithArguments(count)
    }

    /** The symbol from the OpenBLAS library, or from liblapacke when that is where LAPACKE lives. */
    private fun symbol(name: String): MemorySegment = requireNotNull(lookup).find(name).orElse(null)
        ?: lapackeLookup?.find(name)?.orElse(null)
        ?: error("the host OpenBLAS is present but lacks $name")

    private fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle =
        linker.downcallHandle(symbol(name), descriptor, critical)

    /** Looked up the same way as [symbol] but tolerating absence, for a routine koblas can do without. */
    private fun optionalHandle(name: String, descriptor: FunctionDescriptor): MethodHandle? {
        val found = requireNotNull(lookup).find(name).orElse(null) ?: lapackeLookup?.find(name)?.orElse(null)
        return found?.let { linker.downcallHandle(it, descriptor, critical) }
    }

    private fun voidOf(vararg layouts: MemoryLayout) = FunctionDescriptor.ofVoid(*layouts)
    private fun intOf(vararg layouts: MemoryLayout) = FunctionDescriptor.of(JAVA_INT, *layouts)

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

    /** Pins [values] for the call. Nothing is copied, and the address is valid only for that call. */
    fun seg(values: DoubleArray): MemorySegment = MemorySegment.ofArray(values)

    /** Pins [values] for the call. */
    fun seg(values: IntArray): MemorySegment = MemorySegment.ofArray(values)
}
