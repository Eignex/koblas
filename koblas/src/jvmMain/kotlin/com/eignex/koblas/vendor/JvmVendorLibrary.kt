package com.eignex.koblas.vendor

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/**
 * One opened vendor library and the symbols bound out of it.
 *
 * The lookup is the library's own, never the process-global one. That distinction is the whole point on a host
 * with more than one BLAS loaded: `cblas_dgemm` resolves in several of them, and a default lookup would hand
 * back whichever the dynamic loader happened to bind first, so a run labelled oneMKL could be executing
 * OpenBLAS. Resolving against the handle returned for a named file means the label and the code agree.
 *
 * Handles are bound once and held for the process. There is no per-call lookup, and the [Arena.global] lifetime
 * means no handle can outlive its library.
 *
 * Every handle is bound without [Linker.Option.critical]. A vendor BLAS call is not a bounded leaf: it can run
 * for a long time, allocate, and coordinate its own worker threads, and a thread inside a critical downcall
 * holds off a safepoint until it returns. The cost is that operands are copied into native memory rather than
 * pinned in place, which is real and is reported as part of the route rather than hidden.
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
     * The file the key symbol actually came from, asked of the dynamic loader rather than assumed.
     *
     * The candidate that opened is not the same thing. A bare soname matches a library already loaded into the
     * process by some earlier absolute-path open, so the name Koblas asked for can resolve without that name
     * being findable on its own, and two runs can record different strings for one file depending only on what
     * ran first. Asking the loader which object a resolved symbol belongs to gives the file that will actually
     * execute, which is what a report naming a vendor needs to stand on.
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
     * Selects oneMKL's sequential layer, which is a condition of loading it rather than a thread setting.
     *
     * oneMKL resolves its threading layer on first use and defaults to the Intel-threaded one, which needs an
     * OpenMP runtime that a plain oneMKL install does not ship: on a host without `libiomp5` the first call
     * fails with an undefined `omp_get_num_procs` rather than computing anything. Naming the sequential layer
     * at load is what makes the library usable, and it is the layer that matches a library with no threading
     * of its own to coordinate with. Nothing here is configurable and no thread count is ever set.
     */
    private fun selectSequentialLayer() {
        // Resolved to a local first: a safe call makes the site's return type boxed, which invokeExact rejects.
        val handle = handleOrNull("MKL_Set_Threading_Layer", FunctionDescriptor.of(JAVA_INT, JAVA_INT)) ?: return
        handle.invokeExact(MKL_SEQUENTIAL) as Int
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
         * Opens the first candidate of [vendor] that loads and exports every required CBLAS symbol, or null.
         *
         * A library that opens but is missing part of the surface is rejected rather than half-bound, so a
         * partial or mismatched install fails here instead of at the first call that needs the missing piece.
         */
        fun open(vendor: Vendor): JvmVendorLibrary? {
            val linker = try {
                Linker.nativeLinker()
            } catch (_: UnsupportedOperationException) {
                return null
            }
            for (candidate in vendor.resolvedCandidates(System.getProperty("user.home"))) {
                val lookup = try {
                    SymbolLookup.libraryLookup(candidate, Arena.global())
                } catch (_: IllegalArgumentException) {
                    continue // not on this machine
                } catch (_: UnsatisfiedLinkError) {
                    continue // present but unloadable
                }
                val library = JvmVendorLibrary(vendor, candidate, lookup, linker)
                if (VendorOperation.entries.any { it.required && !library.exports(it.entryPoint) }) continue
                library.selectSequentialLayer()
                return library
            }
            return null
        }
    }
}
