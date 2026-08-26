package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.internal.host.FfmLibrary.Companion.intOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.pointerOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.voidOf
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

/**
 * The bridge koblas builds over HiGHS's HFactor. Indices are 32-bit throughout, which is what HighsInt is
 * unless HiGHS is built for 64, so koblas's own CSC arrays and the vectors a solve carries cross without a
 * widening copy.
 */
internal class HfactorCalls(config: HfactorConfig) {
    private val library: FfmLibrary by lazy {
        FfmLibrary.open(
            config.libraryPath?.let(::listOf) ?: HFACTOR_SONAMES,
            "koblas_hfactor_create",
            "HFactor",
        )
    }
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val create: MethodHandle,
        val free: MethodHandle,
        val build: MethodHandle,
        val ftran: MethodHandle,
        val btran: MethodHandle,
        val update: MethodHandle,
        val updateCount: MethodHandle,
        val fill: MethodHandle,
        val pivotRange: MethodHandle,
    )

    /** Whether the library resolved and exports every entry point this binding calls. */
    val available: Boolean get() = handles != null

    private fun bindAll(): Handles? {
        if (!library.present) return null
        return Handles(
            create = library.handleOrNull(
                "koblas_hfactor_create",
                pointerOf(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS),
            ) ?: return null,
            free = library.handleOrNull("koblas_hfactor_free", voidOf(ADDRESS)) ?: return null,
            build = library.handleOrNull("koblas_hfactor_build", intOf(ADDRESS, ADDRESS)) ?: return null,
            ftran = library.handleOrNull(
                "koblas_hfactor_ftran",
                intOf(ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_DOUBLE),
            ) ?: return null,
            btran = library.handleOrNull(
                "koblas_hfactor_btran",
                intOf(ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_DOUBLE),
            ) ?: return null,
            update = library.handleOrNull(
                "koblas_hfactor_update",
                intOf(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT),
            ) ?: return null,
            updateCount = library.handleOrNull("koblas_hfactor_update_count", intOf(ADDRESS)) ?: return null,
            fill = library.handleOrNull("koblas_hfactor_fill", intOf(ADDRESS)) ?: return null,
            pivotRange = library.handleOrNull(
                "koblas_hfactor_pivot_range",
                voidOf(ADDRESS, ADDRESS, ADDRESS),
            ) ?: return null,
        )
    }

    private fun handlesOrThrow(): Handles = checkNotNull(handles) { "HFactor is not available" }

    /**
     * Sets a factorization up over the CSC arrays of the matrix every later basis draws its columns from.
     * Returns null when the library will not take it.
     */
    fun create(rows: Int, cols: Int, colPtr: IntArray, rowIdx: IntArray, values: DoubleArray): MemorySegment? {
        val h = handlesOrThrow()
        val handle = h.create.invokeExact(
            rows,
            cols,
            MemorySegment.ofArray(colPtr),
            MemorySegment.ofArray(rowIdx),
            MemorySegment.ofArray(values),
        ) as MemorySegment
        return handle.takeUnless { it.address() == 0L }
    }

    fun free(handle: MemorySegment) {
        handlesOrThrow().free.invokeExact(handle) as Unit
    }

    /** Factorizes the basis of [basicIndex], answering 0 or the rank deficiency HFactor found. */
    fun build(handle: MemorySegment, basicIndex: IntArray): Int =
        handlesOrThrow().build.invokeExact(handle, MemorySegment.ofArray(basicIndex)) as Int

    /**
     * Solves in place over the vector's own storage, `Bᵀ x = b` when [transpose]. [indices] and [values]
     * are the seam's arrays, handed over as they lie: the call is critical, so they are pinned rather than
     * copied. Returns the solution's nonzero count.
     */
    fun solve(
        handle: MemorySegment,
        count: Int,
        indices: IntArray,
        values: DoubleArray,
        expectedDensity: Double,
        transpose: Boolean,
    ): Int {
        val h = handlesOrThrow()
        val call = if (transpose) h.btran else h.ftran
        return call.invokeExact(
            handle,
            count,
            MemorySegment.ofArray(indices),
            MemorySegment.ofArray(values),
            expectedDensity,
        ) as Int
    }

    /** One Forrest-Tomlin update; see [HfactorUpdate] for what it answers. */
    fun update(handle: MemorySegment, pivotRow: Int, entering: Int, reuseSpike: Boolean, reusePivotEta: Boolean): Int =
        handlesOrThrow().update.invokeExact(
            handle,
            pivotRow,
            entering,
            if (reuseSpike) 1 else 0,
            if (reusePivotEta) 1 else 0,
        ) as Int

    fun updateCount(handle: MemorySegment): Int = handlesOrThrow().updateCount.invokeExact(handle) as Int

    /** The fill the bridge tracks, which reads a counter and so is fit to be read every iteration. */
    fun fill(handle: MemorySegment): Int = handlesOrThrow().fill.invokeExact(handle) as Int

    /**
     * The pivot magnitudes into [range] as smallest then largest. HFactor hands its factors out only by
     * copy, so this duplicates every L and U array; it is for a caller asking what the factors are worth
     * rather than for one pacing its rebuilds, which reads [fill].
     */
    fun pivotRange(handle: MemorySegment, range: DoubleArray) {
        val h = handlesOrThrow()
        Arena.ofConfined().use { arena ->
            val smallest = arena.allocate(JAVA_DOUBLE)
            val largest = arena.allocate(JAVA_DOUBLE)
            h.pivotRange.invokeExact(handle, smallest, largest) as Unit
            range[0] = smallest.get(JAVA_DOUBLE, 0)
            range[1] = largest.get(JAVA_DOUBLE, 0)
        }
    }
}
