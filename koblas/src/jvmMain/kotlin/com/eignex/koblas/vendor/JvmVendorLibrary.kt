package com.eignex.koblas.vendor

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
    /** Which vendor this library is. */
    val vendor: Vendor,
    /** The file that opened, which is the evidence for the vendor label. */
    val path: String,
    private val lookup: SymbolLookup,
    private val linker: Linker,
) {
    /** The library's own version string, or a note that it reports none. */
    val version: String by lazy { readVersion() }

    /**
     * File reported by the dynamic loader for the key symbol. A requested soname can resolve to an
     * already-loaded absolute path, so the candidate name alone does not identify the executing file.
     */
    val resolvedFile: String by lazy { symbolOwner(vendor.keySymbol) ?: path }

    private fun symbolOwner(symbol: String): String? {
        val address = lookup.find(symbol).orElse(null) ?: return null
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

    /**
     * Configure one compute thread before any arithmetic.
     *
     * oneMKL must select its sequential layer before first use to avoid resolving a threaded layer
     * that requires an unavailable OpenMP runtime. Thread count and dynamic expansion are fixed too.
     * [confirmSingleThread] checks the result after the ABI probe; configuration precedes the probe
     * because the probe itself can resolve the threading layer.
     */
    private fun enforceSingleThread() {
        // Every call below sits in statement position with its handle in a local. invokeExact converts
        // nothing, and both a safe call and a lambda's trailing expression give the site a boxed return type
        // that will not match a void descriptor.
        if (vendor.threadControl == ThreadControl.Environment) limitAccelerateThreads()
        if (vendor.threadControl == ThreadControl.Mkl) {
            val layer = handleOrNull("MKL_Set_Threading_Layer", FunctionDescriptor.of(JAVA_INT, JAVA_INT))
            if (layer != null) layer.invokeExact(MKL_SEQUENTIAL) as Int
            val dynamic = handleOrNull("MKL_Set_Dynamic", FunctionDescriptor.ofVoid(JAVA_INT))
            if (dynamic != null) dynamic.invokeExact(0)
        }
        val setter = vendor.threadControl.setter ?: return
        val handle = handleOrNull(setter, FunctionDescriptor.ofVoid(JAVA_INT)) ?: return
        handle.invokeExact(1)
    }

    /**
     * Set Accelerate's process thread limit before arithmetic through libc `setenv`.
     * [ACCELERATE_THREAD_LIMIT] is its available control; overwrite inherited values so they cannot
     * weaken the single-thread invariant.
     */
    private fun limitAccelerateThreads() {
        val address = linker.defaultLookup().find("setenv").orElse(null) ?: return
        val handle = linker.downcallHandle(
            address,
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT),
        )
        Arena.ofConfined().use { arena ->
            val name = arena.allocateFrom(ACCELERATE_THREAD_LIMIT)
            val value = arena.allocateFrom("1")
            handle.invokeExact(name, value, 1) as Int
        }
    }

    /** Set once at load, after [enforceSingleThread], and never again. */
    var threadEvidence: ThreadEvidence = ThreadEvidence.Unconfirmed
        private set

    /**
     * Read back the thread count after [enforceSingleThread]. A library still reporting multiple
     * threads is rejected before arithmetic.
     */
    fun confirmSingleThread(): Boolean {
        val control = vendor.threadControl
        val getter = control.getter
        if (getter == null) {
            // Accelerate, held through the environment and unable to answer. Declared, not discovered.
            threadEvidence = ThreadEvidence.Unconfirmed
            return true
        }
        val handle = handleOrNull(getter, FunctionDescriptor.of(JAVA_INT))
        if (handle == null) {
            // A build of a known vendor without that vendor's control is not one this code knows; see
            // ThreadControl.absenceMeansSerial for the one reading of an absent control that is positive.
            if (!control.absenceMeansSerial) return false
            threadEvidence = ThreadEvidence.Confirmed
            return true
        }
        if ((handle.invokeExact() as Int) != 1) return false
        threadEvidence = ThreadEvidence.Confirmed
        return true
    }

    /**
     * Check ABI compatibility through the build's integer-width declaration; see [declaresWideIntegers].
     * A known-answer arithmetic probe separately rejects libraries that bind but cannot execute.
     */
    fun verifiedAbi(): Boolean {
        if (declaresWideIntegers(version)) return false
        if (!declaredIntegerWidthMatches()) return false
        return probeDot() && probeGemm()
    }

    /** BLIS answers the integer-width question directly; see [BLIS_INTEGER_WIDTH]. */
    private fun declaredIntegerWidthMatches(): Boolean {
        val handle = handleOrNull(BLIS_INTEGER_WIDTH, FunctionDescriptor.of(JAVA_INT)) ?: return true
        return (handle.invokeExact() as Int) == LP64_INTEGER_BITS
    }

    private fun probeDot(): Boolean {
        val handle = handleOrNull(
            BlasOperation.Dot.entryPoint,
            FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
        ) ?: return false
        Arena.ofConfined().use { arena ->
            val x = arena.allocateFrom(JAVA_DOUBLE, *AbiProbe.x)
            val y = arena.allocateFrom(JAVA_DOUBLE, *AbiProbe.y)
            return (handle.invokeExact(AbiProbe.x.size, x, 1, y, 1) as Double) == AbiProbe.DOT
        }
    }

    private fun probeGemm(): Boolean {
        val handle = handleOrNull(
            BlasOperation.Gemm.entryPoint,
            FunctionDescriptor.ofVoid(
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
                ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ),
        ) ?: return false
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

    /** Whether [name] is exported. */
    fun exports(name: String): Boolean = lookup.find(name).isPresent

    /** A non-critical handle for [name], or null when the library does not export it. */
    fun handleOrNull(name: String, descriptor: FunctionDescriptor): MethodHandle? {
        val address = lookup.find(name).orElse(null) ?: return null
        return linker.downcallHandle(address, descriptor)
    }

    /** A non-critical handle for [name], which the library is required to export. */
    fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle =
        checkNotNull(handleOrNull(name, descriptor)) { "${vendor.vendorName} at $path does not export $name" }

    private fun readVersion(): String {
        mklVersion()?.let { return it }
        stringVersion("openblas_get_config")?.let { return it }
        stringVersion("bli_info_get_version_str")?.let { return it }
        return if (vendor == Vendor.Accelerate) "system framework" else "unreported"
    }

    /** oneMKL writes its version into a caller-supplied buffer rather than returning a pointer. */
    private fun mklVersion(): String? {
        val handle = handleOrNull(
            "MKL_Get_Version_String",
            FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT),
        ) ?: return null
        Arena.ofConfined().use { arena ->
            val buffer = arena.allocate(VERSION_BUFFER.toLong())
            handle.invokeExact(buffer, VERSION_BUFFER)
            return buffer.getString(0).trim().ifEmpty { null }
        }
    }

    /** OpenBLAS and BLIS both return a pointer to a static string. */
    private fun stringVersion(symbol: String): String? {
        val handle = handleOrNull(symbol, FunctionDescriptor.of(ADDRESS)) ?: return null
        val address = handle.invokeExact() as MemorySegment
        if (address.equals(MemorySegment.NULL)) return null
        return address.reinterpret(VERSION_BUFFER.toLong()).getString(0).trim().ifEmpty { null }
    }

    companion object {
        private const val VERSION_BUFFER = 256

        /** `MKL_THREADING_SEQUENTIAL`. */
        private const val MKL_SEQUENTIAL = 1

        /** `Dl_info` is four pointers, of which only the first, `dli_fname`, is read. */
        private const val DL_INFO_BYTES = 32L
        private const val PATH_BYTES = 4096L

        /**
         * Open the first [vendor] candidate with every required CBLAS symbol, or null. Reject partial
         * installs before arithmetic; the ABI and single-thread checks must also succeed.
         */
        fun open(vendor: Vendor): JvmVendorLibrary? {
            val linker = try {
                Linker.nativeLinker()
            } catch (_: UnsupportedOperationException) {
                return null
            }
            for (candidate in vendor.resolvedCandidates(System.getProperty("user.home"))) {
                attempt(vendor, candidate, linker)?.let { return it }
            }
            return null
        }

        private fun attempt(vendor: Vendor, candidate: String, linker: Linker): JvmVendorLibrary? {
            val lookup = try {
                SymbolLookup.libraryLookup(candidate, Arena.global())
            } catch (_: IllegalArgumentException) {
                return null // not on this machine
            } catch (_: UnsatisfiedLinkError) {
                return null // present but unloadable
            }
            val library = JvmVendorLibrary(vendor, candidate, lookup, linker)
            if (missingRequiredSymbols(library::exports).isNotEmpty()) return null
            // Ordered: the single-thread configuration is established before the probe, because the probe is
            // arithmetic and oneMKL resolves its threading layer on the first call that reaches it.
            library.enforceSingleThread()
            if (!library.verifiedAbi()) return null
            if (!library.confirmSingleThread()) return null
            return library
        }
    }
}
