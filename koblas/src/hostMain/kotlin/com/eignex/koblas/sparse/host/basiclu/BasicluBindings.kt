@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.internal.host.openNativeLibrary
import kotlinx.cinterop.*
import platform.posix.dlsym
import platform.posix.memset

private typealias Obj = CPointer<ByteVar>?
private typealias Longs = CPointer<LongVar>?
private typealias Doubles = CPointer<DoubleVar>?

/** The bytes `struct basiclu_object` occupies: ten pointers, then `nzlhs` and `realloc_factor`. */
internal const val BASICLU_OBJECT_BYTES = 96

/** `xstore` is the second field, so the store pointer sits one pointer into the object. */
private const val STORE_OFFSET = 8L

private const val PROBE_DIMENSION = 2L
internal const val BASICLU_FORWARD: Byte = 'n'.code.toByte()
internal const val BASICLU_TRANSPOSED: Byte = 't'.code.toByte()

/** The 64-bit `lu_int` object API BASICLU exports, the width its own header fixes for every build. */
internal class BasicluFunctions(private val lib: COpaquePointer) {
    private fun required(name: String): COpaquePointer =
        dlsym(lib, name) ?: error("a BASICLU is present but lacks $name")

    val initialize = required("basiclu_obj_initialize").reinterpret<CFunction<(Obj, Long) -> Long>>()

    val factorize = required("basiclu_obj_factorize")
        .reinterpret<CFunction<(Obj, Longs, Longs, Longs, Doubles) -> Long>>()

    val solveDense = required("basiclu_obj_solve_dense")
        .reinterpret<CFunction<(Obj, Doubles, Doubles, Byte) -> Long>>()

    val solveForUpdate = required("basiclu_obj_solve_for_update")
        .reinterpret<CFunction<(Obj, Long, Longs, Doubles, Byte, Long) -> Long>>()

    val update = required("basiclu_obj_update").reinterpret<CFunction<(Obj, Double) -> Long>>()

    val free = required("basiclu_obj_free").reinterpret<CFunction<(Obj) -> Unit>>()
}

/** Opens the host BASICLU and decides whether it answers in the width these bindings declare. */
internal class BasicluLoader(config: BasicluConfig) {
    private val library: COpaquePointer? by lazy {
        openNativeLibrary(config.libraryPath?.let(::listOf) ?: BASICLU_SONAMES, "basiclu_obj_initialize")
    }

    val functions: BasicluFunctions? by lazy { library?.let(::BasicluFunctions) }

    /**
     * A factorization the library has to get right, not a symbol lookup. HiGHS builds BASICLU with 32-bit
     * `lu_int` and exports these same names, and `xstore` cannot tell the two apart, being doubles at the
     * same offset in both. Every number the probe hands over is small enough that a 32-bit build reading
     * these 64-bit arrays still sees indices inside them, so a mismatched provider answers wrongly rather
     * than reading out of bounds.
     */
    val available: Boolean by lazy { functions?.let(::solvesProbeCorrectly) ?: false }

    private fun solvesProbeCorrectly(f: BasicluFunctions): Boolean {
        val obj = allocateBasicluObject()
        try {
            if (f.initialize(obj, PROBE_DIMENSION) != BasicluStatus.OK) return false
            val solution = DoubleArray(PROBE_DIMENSION.toInt())
            val factored = longArrayOf(0L, 1L).usePinned { begin ->
                longArrayOf(1L, 2L).usePinned { end ->
                    longArrayOf(0L, 1L).usePinned { rows ->
                        doubleArrayOf(2.0, 4.0).usePinned { values ->
                            f.factorize(
                                obj,
                                begin.addressOf(0),
                                end.addressOf(0),
                                rows.addressOf(0),
                                values.addressOf(0),
                            )
                        }
                    }
                }
            }
            if (factored != BasicluStatus.OK) return false
            val solved = doubleArrayOf(2.0, 4.0).usePinned { rhs ->
                solution.usePinned { lhs ->
                    f.solveDense(obj, rhs.addressOf(0), lhs.addressOf(0), BASICLU_FORWARD)
                }
            }
            return solved == BasicluStatus.OK && solution[0] == 1.0 && solution[1] == 1.0
        } finally {
            f.free(obj)
            nativeHeap.free(obj)
        }
    }
}

/**
 * A zeroed `struct basiclu_object`. `nativeHeap` does not clear what it hands out, and
 * `basiclu_obj_initialize` reads the object before it writes it, as the `calloc` a C caller would use says.
 */
internal fun allocateBasicluObject(): CPointer<ByteVar> {
    val obj = nativeHeap.allocArray<ByteVar>(BASICLU_OBJECT_BYTES)
    memset(obj, 0, BASICLU_OBJECT_BYTES.convert())
    return obj
}

/** BASICLU owns the store and reports no length for it, so it is read through its own pointer. */
internal fun basicluStatistic(obj: CPointer<ByteVar>, position: Int): Double {
    val store = (obj + STORE_OFFSET)!!.reinterpret<CPointerVar<DoubleVar>>().pointed.value ?: return 0.0
    return store[position]
}
