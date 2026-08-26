package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.internal.host.FfmLibrary.Companion.longOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.voidOf
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

/** The 64-bit `lu_int` object API BASICLU exports, the width its own header fixes for every build. */
internal class BasicluCalls(private val config: BasicluConfig) {
    private val library: FfmLibrary by lazy {
        FfmLibrary.open(config.libraryPath?.let(::listOf) ?: BASICLU_SONAMES, "basiclu_obj_initialize", "BASICLU")
    }
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val initialize: MethodHandle,
        val factorize: MethodHandle,
        val solveDense: MethodHandle,
        val solveForUpdate: MethodHandle,
        val update: MethodHandle,
        val free: MethodHandle,
    )

    /**
     * Whether the library resolved and answers in the width this binding declares. A library built for
     * 32-bit `lu_int`, which is what HiGHS's vendored copy is, exports these same names and reads the wrong
     * halves of every index it is handed, so availability is a factorization it has to get right rather
     * than a lookup. `xstore` cannot settle it: that array is doubles at the same offset in both builds.
     */
    val available: Boolean by lazy { handles?.let(::answersProbeCorrectly) ?: false }

    private fun bindAll(): Handles? {
        if (!library.present) return null
        return Handles(
            initialize = library.handleOrNull("basiclu_obj_initialize", longOf(ADDRESS, JAVA_LONG)) ?: return null,
            factorize = library.handleOrNull(
                "basiclu_obj_factorize",
                longOf(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
            ) ?: return null,
            solveDense = library.handleOrNull(
                "basiclu_obj_solve_dense",
                longOf(ADDRESS, ADDRESS, ADDRESS, JAVA_BYTE),
            ) ?: return null,
            solveForUpdate = library.handleOrNull(
                "basiclu_obj_solve_for_update",
                longOf(ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_BYTE, JAVA_LONG),
            ) ?: return null,
            update = library.handleOrNull("basiclu_obj_update", longOf(ADDRESS, JAVA_DOUBLE)) ?: return null,
            free = library.handleOrNull("basiclu_obj_free", voidOf(ADDRESS)) ?: return null,
        )
    }

    /**
     * Factorizes `diag(2, 4)` and solves it back to `(1, 1)`, which exercises the index arrays where the
     * integer width lives. Every number handed over is small enough that a 32-bit build reading these
     * 64-bit arrays still sees indices inside them, so a mismatched provider answers wrongly rather than
     * reading out of bounds. The solve is exact, so the comparison can be.
     */
    private fun answersProbeCorrectly(h: Handles): Boolean = Arena.ofConfined().use { arena ->
        val obj = arena.allocate(OBJECT_BYTES)
        if (h.initialize.invokeExact(obj, PROBE_DIMENSION) as Long != BasicluStatus.OK) return false
        try {
            val begin = arena.allocateFrom(JAVA_LONG, 0L, 1L)
            val end = arena.allocateFrom(JAVA_LONG, 1L, 2L)
            val rows = arena.allocateFrom(JAVA_LONG, 0L, 1L)
            val values = arena.allocateFrom(JAVA_DOUBLE, 2.0, 4.0)
            if (h.factorize.invokeExact(obj, begin, end, rows, values) as Long != BasicluStatus.OK) return false
            val rhs = arena.allocateFrom(JAVA_DOUBLE, 2.0, 4.0)
            val solution = arena.allocate(JAVA_DOUBLE, PROBE_DIMENSION)
            if (h.solveDense.invokeExact(obj, rhs, solution, FORWARD) as Long != BasicluStatus.OK) return false
            solution.get(JAVA_DOUBLE, 0) == 1.0 &&
                solution.get(JAVA_DOUBLE, Double.SIZE_BYTES.toLong()) == 1.0
        } finally {
            h.free.invokeExact(obj) as Unit
        }
    }

    private fun handlesOrThrow(): Handles = checkNotNull(handles) { "BASICLU is not available" }

    /**
     * Initializes an object for an `n` by `n` basis, or returns null when BASICLU cannot. The arena is
     * shared rather than confined because the factorization holding it is freed from a cleaner thread.
     */
    fun create(n: Int): BasicluObject? {
        val h = handlesOrThrow()
        val arena = Arena.ofShared()
        val obj = arena.allocate(OBJECT_BYTES)
        if (h.initialize.invokeExact(obj, n.toLong()) as Long != BasicluStatus.OK) {
            arena.close()
            return null
        }
        return BasicluObject(n, arena, obj, arena.allocate(JAVA_DOUBLE, n.toLong()))
    }

    /** Factorizes the CSC arrays of a square matrix into [target], reporting BASICLU's own status. */
    fun factorize(target: BasicluObject, colPtr: IntArray, rowIdx: IntArray, values: DoubleArray): Long {
        val h = handlesOrThrow()
        return Arena.ofConfined().use { arena ->
            val begin = arena.allocateFrom(JAVA_LONG, *LongArray(target.n) { colPtr[it].toLong() })
            val end = arena.allocateFrom(JAVA_LONG, *LongArray(target.n) { colPtr[it + 1].toLong() })
            val rows = arena.allocateFrom(JAVA_LONG, *LongArray(rowIdx.size) { rowIdx[it].toLong() })
            val entries = arena.allocateFrom(JAVA_DOUBLE, *values)
            h.factorize.invokeExact(target.obj, begin, end, rows, entries) as Long
        }
    }

    /**
     * Solves in place through [target]'s own scratch, `Bᵀ x = b` when [transpose].
     *
     * The right-hand side crosses as a segment over the caller's own array, which a critical downcall reads
     * where it lies, rather than through an arena that cost a copy each way and an allocation per solve on
     * the routine a simplex calls twice an iteration. BASICLU writes its answer to a second buffer, so that
     * one is still copied back.
     */
    fun solve(target: BasicluObject, values: DoubleArray, transpose: Boolean): Long {
        val h = handlesOrThrow()
        val rhs = MemorySegment.ofArray(values)
        val status = h.solveDense.invokeExact(target.obj, rhs, target.solution, transposeFlag(transpose)) as Long
        if (status == BasicluStatus.OK) {
            MemorySegment.copy(target.solution, JAVA_DOUBLE, 0, values, 0, values.size)
        }
        return status
    }

    /**
     * Prepares and applies a Forrest-Tomlin update for the entering column [indices] and [values] describe,
     * taking the place of [column]. BASICLU wants that column solved for first and then the leaving row
     * named through a transposed solve, both before the update itself, so the three calls belong together.
     */
    fun replaceColumn(target: BasicluObject, column: Int, indices: IntArray, values: DoubleArray): Long {
        val h = handlesOrThrow()
        return Arena.ofConfined().use { arena ->
            val rows = arena.allocateFrom(JAVA_LONG, *LongArray(indices.size) { indices[it].toLong() })
            val entries = arena.allocateFrom(JAVA_DOUBLE, *values)
            val forward = h.solveForUpdate.invokeExact(
                target.obj,
                indices.size.toLong(),
                rows,
                entries,
                FORWARD,
                0L,
            ) as Long
            if (forward != BasicluStatus.OK) return@use forward
            val leaving = arena.allocateFrom(JAVA_LONG, column.toLong())
            val transposed = h.solveForUpdate.invokeExact(
                target.obj,
                1L,
                leaving,
                MemorySegment.NULL,
                TRANSPOSED,
                0L,
            ) as Long
            if (transposed != BasicluStatus.OK) return@use transposed
            h.update.invokeExact(target.obj, 0.0) as Long
        }
    }

    /** A statistic from [target]'s `xstore`, which BASICLU keeps as doubles throughout. */
    fun statistic(target: BasicluObject, position: Int): Double =
        store(target.obj).get(JAVA_DOUBLE, position * Double.SIZE_BYTES.toLong())

    /** Bound through a local, since a null-safe call would compile to a boxed return `invokeExact` rejects. */
    fun free(target: BasicluObject) {
        val h = handles ?: return
        h.free.invokeExact(target.obj) as Unit
    }

    private fun transposeFlag(transpose: Boolean): Byte = if (transpose) TRANSPOSED else FORWARD

    private companion object {
        /** Ten pointers, then `nzlhs` and `realloc_factor`, which is what a 64-bit `lu_int` build lays out. */
        const val OBJECT_BYTES = 96L

        /** `xstore` is the second field, so the store pointer sits one pointer in. */
        const val STORE_OFFSET = 8L

        const val PROBE_DIMENSION = 2L
        const val FORWARD: Byte = 'n'.code.toByte()
        const val TRANSPOSED: Byte = 't'.code.toByte()

        /** BASICLU owns the store and reports no length for it, so it is re-sized before it is read. */
        fun store(obj: MemorySegment): MemorySegment = obj.get(ADDRESS, STORE_OFFSET).reinterpret(Long.MAX_VALUE)
    }
}

/** One BASICLU object, its arena, and the destination its dense solves write. */
internal class BasicluObject(val n: Int, val arena: Arena, val obj: MemorySegment, val solution: MemorySegment)
