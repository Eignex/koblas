package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.host.*
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/**
 * Vendor library and symbols bound from its own handle.
 *
 * A process-global lookup could resolve another loaded BLAS, making the vendor label inaccurate.
 * Handles are bound once and share the library's [Arena.global] lifetime.
 *
 * Downcalls omit [Linker.Option.critical]: vendor routines can allocate, run for long periods and
 * coordinate worker threads. Critical calls would hold off safepoints. Native operand copies are
 * required and reported in the route.
 */
internal class JvmVendorLibrary private constructor(
    override val vendor: Vendor,
    override val candidate: String,
    private val lookup: SymbolLookup,
    private val linker: Linker,
) : AdmissionCalls {
    override val version: String by lazy { readVersion() }

    /**
     * File reported by the dynamic loader for the key symbol. A requested soname can resolve to an
     * already-loaded absolute path, so the candidate name alone does not identify the executing file.
     */
    val resolvedFile: String by lazy { symbolOwner(vendor.keySymbol) ?: candidate }

    private fun symbolOwner(symbol: String): String? {
        val address = lookup.find(symbol).orElse(null) ?: return null
        if (hostPlatform().operatingSystem == OperatingSystem.Windows) return windowsSymbolOwner(address)
        val dladdr = linker.defaultLookup().find("dladdr").orElse(null) ?: return null
        val handle = linker.downcallHandle(dladdr, FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS))
        Arena.ofConfined().use { arena ->
            val info = arena.allocate(DL_INFO_BYTES)
            val found = handle.invokeExact(address, info) as Int
            if (found == 0) return null
            val name = info.get(ADDRESS, 0L)
            if (name.equals(MemorySegment.NULL)) return null
            return name.reinterpret(PATH_BYTES).getString(0).ifEmpty { null }
        }
    }

    /** Resolve the actual DLL, including a forwarded export, through the Windows loader. */
    private fun windowsSymbolOwner(address: MemorySegment): String? {
        val kernel = SymbolLookup.libraryLookup("kernel32.dll", Arena.global())
        val moduleForAddress = linker.downcallHandle(
            kernel.find("GetModuleHandleExW").orElseThrow(),
            FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS),
        )
        val moduleFile = linker.downcallHandle(
            kernel.find("GetModuleFileNameW").orElseThrow(),
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT),
        )
        Arena.ofConfined().use { arena ->
            val owner = arena.allocate(ADDRESS)
            // FROM_ADDRESS | UNCHANGED_REFCOUNT keeps this query from retaining another module handle.
            if ((moduleForAddress.invokeExact(WINDOWS_MODULE_FLAGS, address, owner) as Int) == 0) return null
            val module = owner.get(ADDRESS, 0L)
            val path = arena.allocate(WINDOWS_PATH_CHARACTERS * 2L, 2L)
            val length = moduleFile.invokeExact(module, path, WINDOWS_PATH_CHARACTERS) as Int
            if (length <= 0 || length >= WINDOWS_PATH_CHARACTERS) return null
            return path.getString(0L, Charsets.UTF_16LE)
        }
    }

    // Every invokeExact below takes its exact type from a cast applied directly to it, or from statement position
    // for a void call, because it converts nothing. A safe call, or a void call as a lambda's trailing
    // expression, would give the site a boxed return type that matches no descriptor.

    override fun callIntToInt(name: String, argument: Int): Int? {
        val handle = handleOrNull(name, FunctionDescriptor.of(JAVA_INT, JAVA_INT)) ?: return null
        val result = handle.invokeExact(argument) as Int
        return result
    }

    override fun callIntToUnit(name: String, argument: Int) {
        val handle = handleOrNull(name, FunctionDescriptor.ofVoid(JAVA_INT)) ?: return
        handle.invokeExact(argument)
    }

    override fun callToInt(name: String): Int? {
        val handle = handleOrNull(name, FunctionDescriptor.of(JAVA_INT)) ?: return null
        val result = handle.invokeExact() as Int
        return result
    }

    override fun callIntoBuffer(name: String, size: Int): String? {
        val handle = handleOrNull(name, FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT)) ?: return null
        Arena.ofConfined().use { arena ->
            val buffer = arena.allocate(size.toLong())
            handle.invokeExact(buffer, size)
            return buffer.getString(0)
        }
    }

    override fun callToString(name: String): String? {
        val handle = handleOrNull(name, FunctionDescriptor.of(ADDRESS)) ?: return null
        val address = handle.invokeExact() as MemorySegment
        if (address.equals(MemorySegment.NULL)) return null
        return address.reinterpret(VERSION_BUFFER.toLong()).getString(0)
    }

    /** Through libc `setenv`, with overwrite on. */
    override fun setEnvironmentVariable(name: String, value: String) {
        val address = linker.defaultLookup().find("setenv").orElse(null) ?: return
        val handle = linker.downcallHandle(address, FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT))
        Arena.ofConfined().use { arena ->
            handle.invokeExact(arena.allocateFrom(name), arena.allocateFrom(value), 1) as Int
        }
    }

    override fun holdThisThread() {
        val setter = threadSetter ?: return
        setter.invokeExact(1)
        heldThreads.set(true)
    }

    override fun knownAnswersMatch(): Boolean = probeDot() && probeGemm()

    private val threadSetter: MethodHandle? by lazy {
        vendor.threadControl.setter?.let { handleOrNull(it, FunctionDescriptor.ofVoid(JAVA_INT)) }
    }

    private val heldThreads: ThreadLocal<Boolean> = ThreadLocal.withInitial { false }

    /**
     * Apply the one-thread setter on the calling thread, once per thread.
     *
     * OpenMP keeps its thread count per thread, as do current BLIS builds, and an OpenMP build of OpenBLAS
     * resynchronises its count from the calling thread's. A setting made only on the thread that opened the
     * library would leave every other caller with a full team, so each thread takes the setting before its
     * first call.
     */
    fun holdCallingThread() {
        if (!heldThreads.get()) holdThisThread()
    }

    /** Set once at load by [admit], and never again. */
    var threadEvidence: ThreadEvidence = ThreadEvidence.Unconfirmed
        private set

    @Suppress("SpreadOperator") // The small ABI probe copies only once when a library is opened.
    private fun probeDot(): Boolean {
        val handle = handleOrNull(EngineImplementation.Dot.entryPoint, DOT) ?: return false
        Arena.ofConfined().use { arena ->
            val x = arena.allocateFrom(JAVA_DOUBLE, *AbiProbe.x)
            val y = arena.allocateFrom(JAVA_DOUBLE, *AbiProbe.y)
            return (handle.invokeExact(AbiProbe.x.size, x, 1, y, 1) as Double) == AbiProbe.DOT
        }
    }

    @Suppress("SpreadOperator") // The small ABI probe copies only once when a library is opened.
    private fun probeGemm(): Boolean {
        val handle = handleOrNull(EngineImplementation.Gemm.entryPoint, GEMM) ?: return false
        val expected = AbiProbe.operand
        Arena.ofConfined().use { arena ->
            val a = arena.allocateFrom(JAVA_DOUBLE, *AbiProbe.identity)
            val b = arena.allocateFrom(JAVA_DOUBLE, *expected)
            val c = arena.allocate(JAVA_DOUBLE, expected.size.toLong())
            handle.invokeExact(
                Cblas.COL_MAJOR, Cblas.NO_TRANS, Cblas.NO_TRANS, 2, 2, 2, 1.0,
                a, 2, b, 2, 0.0, c, 2,
            )
            return expected.indices.all { c.getAtIndex(JAVA_DOUBLE, it.toLong()) == expected[it] }
        }
    }

    /** A non-critical handle for [name], or null when the library does not export it. */
    fun handleOrNull(name: String, descriptor: FunctionDescriptor): MethodHandle? {
        val address = lookup.find(name).orElse(null) ?: return null
        return linker.downcallHandle(address, descriptor)
    }

    companion object {
        /** `Dl_info` is four pointers, of which only the first, `dli_fname`, is read. */
        private const val DL_INFO_BYTES = 32L
        private const val PATH_BYTES = 4096L
        private const val WINDOWS_PATH_CHARACTERS = 32768
        private const val WINDOWS_MODULE_FLAGS = 6

        /**
         * Open the first [vendor] candidate that passes the LP64 probes and single-thread checks.
         * Missing non-probe exports remain absent capabilities.
         */
        fun open(vendor: Vendor): JvmVendorLibrary? {
            val linker = try {
                Linker.nativeLinker()
            } catch (_: UnsupportedOperationException) {
                return null
            }
            for (candidate in vendor.loadCandidates(
                System.getProperty("user.home"),
                hostPlatform().operatingSystem,
                System.getenv("PATH"),
            )) {
                attempt(vendor, candidate, linker)?.let { return it }
            }
            return null
        }

        private fun attempt(vendor: Vendor, candidate: String, linker: Linker): JvmVendorLibrary? {
            val lookup = try {
                if (hostPlatform().operatingSystem == OperatingSystem.Windows) {
                    windowsLibraryLookup(candidate, linker) ?: return null
                } else {
                    SymbolLookup.libraryLookup(candidate, Arena.global())
                }
            } catch (_: IllegalArgumentException) {
                return null // not on this machine
            } catch (_: UnsatisfiedLinkError) {
                return null // present but unloadable
            } catch (_: IllegalCallerException) {
                return null // native access denied to this module
            }
            val library = JvmVendorLibrary(vendor, candidate, lookup, linker)
            library.threadEvidence = library.admit() ?: return null
            return library
        }
    }
}

private val DOT = FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)
private val PAIR = FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)
private val RANK_TWO = FunctionDescriptor.ofVoid(
    JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
)
private val TRIANGULAR_VECTOR = FunctionDescriptor.ofVoid(
    JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
)
private val GEMM = FunctionDescriptor.ofVoid(
    JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
    ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
)
private val SYMM = FunctionDescriptor.ofVoid(
    JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
    ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
)
private val TRIANGULAR_MATRIX = FunctionDescriptor.ofVoid(
    JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
    ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
)

/**
 * The foreign shape of [operation]'s CBLAS entry point, or null when it is not a foreign call. Routines that
 * share a signature share a descriptor: `dgemmt` has `dgemm`'s, `dsyr2k` has `dsymm`'s and `dsyr2` has `dger`'s.
 *
 * `CBLAS_INDEX` is `size_t` in the reference header and in several vendors, and `int` in others. The low 32 bits
 * agree for every index a 32-bit BLAS integer can address, so binding `idamax` as `int` reads the same value
 * either way and avoids depending on which one this library used.
 */
internal fun descriptorFor(operation: EngineImplementation): FunctionDescriptor? = when (operation) {
    EngineImplementation.Dot -> DOT

    EngineImplementation.Nrm2, EngineImplementation.Asum ->
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT)

    EngineImplementation.Iamax -> FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT)

    EngineImplementation.Axpy ->
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)

    EngineImplementation.Scale -> FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT)

    EngineImplementation.Copy, EngineImplementation.Swap -> PAIR

    EngineImplementation.Rot ->
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE)

    EngineImplementation.Gemv, EngineImplementation.GemvTransposed -> FunctionDescriptor.ofVoid(
        JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
    )

    EngineImplementation.Symv -> FunctionDescriptor.ofVoid(
        JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
    )

    EngineImplementation.Ger, EngineImplementation.Syr2 -> RANK_TWO

    EngineImplementation.Syr -> FunctionDescriptor.ofVoid(
        JAVA_INT,
        JAVA_INT,
        JAVA_INT,
        JAVA_DOUBLE,
        ADDRESS,
        JAVA_INT,
        ADDRESS,
        JAVA_INT,
    )

    EngineImplementation.Trmv, EngineImplementation.TrmvTransposed,
    EngineImplementation.Trsv, EngineImplementation.TrsvTransposed,
    -> TRIANGULAR_VECTOR

    EngineImplementation.Gemm, EngineImplementation.Gemmt -> GEMM

    EngineImplementation.Symm, EngineImplementation.Syr2k -> SYMM

    EngineImplementation.Syrk -> FunctionDescriptor.ofVoid(
        JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
        ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
    )

    EngineImplementation.Trmm, EngineImplementation.Trsm -> TRIANGULAR_MATRIX

    else -> null
}
