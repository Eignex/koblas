package com.eignex.koblas.openblas

import org.bytedeco.javacpp.Loader
import org.bytedeco.openblas.global.openblas
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
import java.nio.file.Path

/**
 * The CBLAS/LAPACKE subset this backend dispatches to, bound with `java.lang.foreign` downcalls rather
 * than JNI.
 *
 * The point is [Linker.Option.critical] with heap access: a Kotlin `DoubleArray` can be handed over as
 * `MemorySegment.ofArray(...)` and is *pinned* for the duration of the call, so nothing is copied. The
 * JNI-based bindings instead materialize a native buffer per call, which is why level-2 routines lost by
 * 3-16x to the portable SIMD kernels — `O(n²)` work cannot amortize an `O(n²)` copy. Level 3 and the
 * factorizations amortize it over `O(n³)` work, but with this binding they do not pay it at all.
 *
 * The natives still ship in the Bytedeco artifacts: [Loader] extracts them on first use and this looks the
 * library up by that path, so no system OpenBLAS installation is required.
 *
 * Pinning blocks relocation of the array while the call runs, which is the trade for zero copy. koblas's
 * calls are leaf work of bounded duration, and level-2 shapes do not reach here at all.
 */
internal object OpenBlasCalls {

    // The CBLAS enums and the LAPACKE layout macro, by their ABI integer values.
    const val ROW_MAJOR = 101
    const val NO_TRANS = 111
    const val TRANS = 112
    const val UPPER = 121
    const val LOWER = 122
    const val NON_UNIT = 131
    const val UNIT = 132
    const val LEFT = 141
    const val RIGHT = 142

    /** Whether the natives loaded and the symbols resolved on this host. */
    val available: Boolean

    private val linker = Linker.nativeLinker()

    // Declares that the downcall may read or write on-heap arrays handed over as segments.
    private val critical = Linker.Option.critical(true)

    private lateinit var lookup: SymbolLookup

    init {
        available = try {
            // Loader.load returns the extracted JNI shim; libopenblas sits beside it.
            val jni = Path.of(Loader.load(openblas::class.java))
            val dir = jni.parent
            val candidates = listOf("libopenblas.so.0", "libopenblas.so", "libopenblas.dylib", "openblas.dll")
            val found = candidates.map(dir::resolve).firstOrNull { it.toFile().exists() }
            if (found == null) {
                false
            } else {
                lookup = SymbolLookup.libraryLookup(found, Arena.global())
                lookup.find("cblas_dgemm").isPresent && lookup.find("LAPACKE_dgetrf").isPresent
            }
        } catch (_: Throwable) { // a missing native for this platform must not crash startup
            false
        }
    }

    private fun handle(name: String, descriptor: FunctionDescriptor): MethodHandle =
        linker.downcallHandle(lookup.find(name).orElseThrow(), descriptor, critical)

    private fun voidOf(vararg layouts: MemoryLayout) = FunctionDescriptor.ofVoid(*layouts)
    private fun intOf(vararg layouts: MemoryLayout) = FunctionDescriptor.of(JAVA_INT, *layouts)

    // Level 3 and the factorization family. Enum arguments are int, their C ABI; lapack_int is 32-bit,
    // matching the default (non-INTERFACE64) OpenBLAS build.
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
