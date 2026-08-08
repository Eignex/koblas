package com.eignex.koblas.hostblas

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
 * The CBLAS/LAPACKE subset this backend dispatches to, bound with `java.lang.foreign` downcalls rather
 * than JNI.
 *
 * The point is `Linker.Option.critical` with heap access: a Kotlin `DoubleArray` can be handed over as
 * `MemorySegment.ofArray(...)` and is *pinned* for the duration of the call, so nothing is copied. The
 * JNI-based bindings instead materialize a native buffer per call, which is why level-2 routines lost by
 * 3-16x to the portable SIMD kernels — `O(n²)` work cannot amortize an `O(n²)` copy. Level 3 and the
 * factorizations amortize it over `O(n³)` work, but with this binding they do not pay it at all.
 *
 * Nothing is bundled: the host's own OpenBLAS is looked up by soname, so this backend is available exactly
 * when a machine has the library installed and absent otherwise, with the portable kernels taking over.
 * CBLAS and LAPACKE are probed separately, because Debian and Ubuntu ship LAPACKE as a package that
 * libopenblas does not pull in.
 *
 * Pinning blocks relocation of the array while the call runs, which is the trade for zero copy. koblas's
 * calls are leaf work of bounded duration, and level-2 shapes do not reach here at all.
 *
 * Calls go through `invokeWithArguments` rather than `invokeExact`: the latter is signature-polymorphic,
 * which Kotlin cannot emit — it boxes the arguments instead, and the native side then reads garbage
 * (observed as a SIGSEGV at any realistic size). The boxing `invokeWithArguments` does costs on the order
 * of 100 ns against calls that do `O(n³)` work, so it is not measurable here.
 */
internal object HostBlasCalls {

    // The CBLAS enums and the LAPACKE layout macro, by their ABI integer values.
    const val COL_MAJOR = 102
    const val NO_TRANS = 111
    const val TRANS = 112
    const val UPPER = 121
    const val LOWER = 122
    const val NON_UNIT = 131
    const val UNIT = 132
    const val LEFT = 141
    const val RIGHT = 142

    /**
     * Whether the host's CBLAS resolved *and* takes the integer width koblas binds — which is all the
     * [com.eignex.koblas.dense.Blas] half needs.
     */
    val available: Boolean

    /** Whether LAPACKE resolved as well; false on a host that ships CBLAS only. */
    val lapackAvailable: Boolean

    /** Why the host BLAS is unusable, or null when it is usable. For diagnostics, not control flow. */
    val unavailableReason: String?

    private val linker = Linker.nativeLinker()

    // Declares that the downcall may read or write on-heap arrays handed over as segments.
    private val critical = Linker.Option.critical(true)

    private var lookup: SymbolLookup? = null

    /** Where LAPACKE lives when the OpenBLAS build does not include it. */
    private var lapackeLookup: SymbolLookup? = null

    /**
     * Sonames rather than paths: the platform loader searches its usual places, so a package-manager
     * install is found without koblas knowing where it landed. Homebrew keeps OpenBLAS keg-only, hence
     * the explicit prefixes on macOS.
     */
    private val SONAMES = listOf(
        "libopenblas.so.0",
        "libopenblas.so",
        "libopenblas.dylib",
        "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
        "/usr/local/opt/openblas/lib/libopenblas.dylib",
        "openblas.dll",
    )

    private val LAPACKE_SONAMES = listOf("liblapacke.so.3", "liblapacke.so", "liblapacke.dylib", "lapacke.dll")

    init {
        val blas = openLibrary(SONAMES)
        lookup = blas
        val resolved = blas != null && blas.find("cblas_dgemm").isPresent
        // An ILP64 build exports these same names and takes 64-bit integers, so resolution cannot tell it
        // apart from the LP64 one these bindings declare. The config string can.
        val ilp64 = resolved && isIlp64OpenBlas(configString(requireNotNull(blas)))
        available = resolved && !ilp64
        unavailableReason = when {
            blas == null -> "libopenblas could not be opened; OpenBLAS does not appear to be installed"
            !resolved -> "libopenblas opened but lacks cblas_dgemm"
            ilp64 -> "libopenblas reports a 64-bit integer interface, which koblas does not bind"
            else -> null
        }
        val lapackeInBlas = available && requireNotNull(blas).find("LAPACKE_dgetrf").isPresent
        // Debian and Ubuntu strip LAPACKE out of their OpenBLAS build and ship it as liblapacke, a
        // package libopenblas does not pull in. Trying the second library is what keeps the LAPACK half
        // native on the most common Linux setup instead of falling back to the portable factorizations.
        val extra = if (available && !lapackeInBlas) openLibrary(LAPACKE_SONAMES) else null
        lapackeLookup = extra
        lapackAvailable = lapackeInBlas || extra?.find("LAPACKE_dgetrf")?.isPresent == true
        // Before any routine can run. An unconfigured OpenBLAS is multithreaded, and its parallel LAPACK
        // overflows a default JVM thread stack: the process dies with SIGSEGV rather than an exception,
        // so this is not something a caller could catch. Configuring at resolution covers both halves,
        // whichever is constructed first.
        if (available) configureThreads()
    }

    /**
     * The library's `openblas_get_config` string, or empty when it does not offer one.
     *
     * The one call made during resolution, and it is a string read rather than arithmetic. A build without
     * the symbol cannot be interrogated, and an empty string reads as "no disqualifying marker" — the
     * assumption koblas makes wherever a library declines to describe itself.
     */
    private fun configString(blas: SymbolLookup): String {
        val symbol = blas.find("openblas_get_config").orElse(null) ?: return ""
        val handle = linker.downcallHandle(symbol, FunctionDescriptor.of(ADDRESS))
        val text = handle.invokeWithArguments() as MemorySegment
        if (text.address() == 0L) return ""
        // The returned pointer carries no length, so it has to be re-sized before the string can be read.
        return text.reinterpret(Long.MAX_VALUE).getString(0)
    }

    /**
     * Pins OpenBLAS to one thread unless told otherwise.
     *
     * `koblas.openblas.threads` opts into a specific count; an explicit `OPENBLAS_NUM_THREADS` is honored
     * by the library itself and left alone. Single-threaded is also the faster configuration at koblas
     * workload sizes, so the safe default costs nothing here.
     */
    private fun configureThreads() {
        val requested = System.getProperty("koblas.openblas.threads")?.toIntOrNull()
        when {
            requested != null -> setThreads(requested)
            System.getenv("OPENBLAS_NUM_THREADS") == null -> setThreads(1)
        }
    }

    /** Opens the first of [names] the platform loader can find, or null when none resolves. */
    private fun openLibrary(names: List<String>): SymbolLookup? = try {
        names.firstNotNullOfOrNull { runCatching { SymbolLookup.libraryLookup(it, Arena.global()) }.getOrNull() }
    } catch (_: Throwable) { // a host without the library must not crash startup
        null
    }

    /**
     * Threads must be configured on *this* instance. The library is resolved by path here, which is not
     * necessarily the handle the JNI presets configured, and an unconfigured OpenBLAS runs multithreaded:
     * its parallel LAPACK path overflows default JVM thread stacks and crashes the process at any
     * realistic size. Single-threaded is also the faster configuration at koblas workload sizes.
     */
    fun setThreads(count: Int) {
        val setter = requireNotNull(lookup).find("openblas_set_num_threads").orElse(null) ?: return
        linker.downcallHandle(setter, FunctionDescriptor.ofVoid(JAVA_INT)).invokeWithArguments(count)
    }

    /** The symbol from the OpenBLAS library, or from liblapacke when that is where LAPACKE lives. */
    private fun symbol(name: String): MemorySegment = requireNotNull(lookup).find(name).orElse(null)
        ?: lapackeLookup?.find(name)?.orElse(null)
        ?: error("the host OpenBLAS is present but lacks $name")

    private fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle =
        linker.downcallHandle(symbol(name), descriptor, critical)

    private fun voidOf(vararg layouts: MemoryLayout) = FunctionDescriptor.ofVoid(*layouts)
    private fun intOf(vararg layouts: MemoryLayout) = FunctionDescriptor.of(JAVA_INT, *layouts)

    // Level 3 and the factorization family. Enum arguments are int, their C ABI; lapack_int is 32-bit,
    // matching the default (non-INTERFACE64) OpenBLAS build.
    // Level 2. Bound because the routing is a threshold now, not a hardcoded delegation: the portable
    // kernels win at every size measured, but the gate has to be able to choose otherwise.
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

    val dpotrs: MethodHandle by lazy {
        handle("LAPACKE_dpotrs", intOf(JAVA_INT, JAVA_BYTE, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT))
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

    // int LAPACKE_dgeqp3(int layout, int m, int n, double* a, int lda, int* jpvt, double* tau)
    val dgeqp3: MethodHandle by lazy {
        handle("LAPACKE_dgeqp3", intOf(JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS))
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

    /** Pins [values] for the call: no copy, and the address is only valid for that call's duration. */
    fun seg(values: DoubleArray): MemorySegment = MemorySegment.ofArray(values)

    /** Pins [values] for the call. */
    fun seg(values: IntArray): MemorySegment = MemorySegment.ofArray(values)
}
