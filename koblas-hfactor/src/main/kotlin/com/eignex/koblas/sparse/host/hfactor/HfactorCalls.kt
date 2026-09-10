package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.hfactor.internal.HfactorLibrary
import com.eignex.koblas.hfactor.internal.HfactorLibrary.Companion.intOf
import com.eignex.koblas.hfactor.internal.HfactorLibrary.Companion.pointerOf
import com.eignex.koblas.hfactor.internal.HfactorLibrary.Companion.voidOf
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

// HFactor uses 32-bit indices by default, matching koblas's CSC arrays and avoiding a widening copy.
internal class HfactorCalls(private val config: HfactorConfig) {
    private val library: HfactorLibrary by lazy {
        HfactorLibrary.open(
            config.libraryPath?.let(::listOf) ?: HFACTOR_SONAMES,
            "koblas_hfactor_create_v2",
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
        // Optional diagnostics symbols do not prevent factorization.
        val buildRepairing: MethodHandle?,
        val snapshot: MethodHandle?,
        val restore: MethodHandle?,
        val snapshotFree: MethodHandle?,
        val refactorizeReason: MethodHandle?,
        val kernel: MethodHandle?,
    )

    val available: Boolean get() = handles != null

    val unavailableReason: String?
        get() {
            if (!library.present) return library.unavailableReason
            if (handles == null) return "the HFactor library is missing a required symbol"
            return null
        }

    private fun bindAll(): Handles? {
        if (!library.present) return null
        return Handles(
            create = library.handleOrNull(
                "koblas_hfactor_create_v2",
                pointerOf(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_INT),
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
            buildRepairing = library.handleOrNull(
                "koblas_hfactor_build_repairing",
                intOf(ADDRESS, ADDRESS, ADDRESS),
            ),
            snapshot = library.handleOrNull("koblas_hfactor_snapshot", pointerOf(ADDRESS)),
            restore = library.handleOrNull("koblas_hfactor_restore", intOf(ADDRESS, ADDRESS)),
            snapshotFree = library.handleOrNull("koblas_hfactor_snapshot_free", voidOf(ADDRESS)),
            refactorizeReason = library.handleOrNull("koblas_hfactor_refactorize_reason", intOf(ADDRESS)),
            kernel = library.handleOrNull("koblas_hfactor_kernel", voidOf(ADDRESS, ADDRESS, ADDRESS)),
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
            config.pivotThreshold,
            config.pivotTolerance,
            config.updateMethod.nativeValue,
        ) as MemorySegment
        return handle.takeUnless { it.address() == 0L }
    }

    fun free(handle: MemorySegment) {
        handlesOrThrow().free.invokeExact(handle) as Unit
    }

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

    fun update(handle: MemorySegment, pivotRow: Int, entering: Int, reuseSpike: Boolean, reusePivotEta: Boolean): Int =
        handlesOrThrow().update.invokeExact(
            handle,
            pivotRow,
            entering,
            if (reuseSpike) 1 else 0,
            if (reusePivotEta) 1 else 0,
        ) as Int

    fun updateCount(handle: MemorySegment): Int = handlesOrThrow().updateCount.invokeExact(handle) as Int

    fun fill(handle: MemorySegment): Int = handlesOrThrow().fill.invokeExact(handle) as Int

    /**
     * Factorizes keeping HFactor's repair, writing the basis it settled on into [repaired] and returning the
     * rank deficiency, or null where the shim predates this entry point.
     */
    @Suppress("SpreadOperator") // FFM copies the caller-owned indices into confined native storage.
    fun buildRepairing(handle: MemorySegment, basicIndex: IntArray, repaired: IntArray): Int? {
        val builder = handlesOrThrow().buildRepairing ?: return null
        Arena.ofConfined().use { arena ->
            val given = arena.allocateFrom(JAVA_INT, *basicIndex)
            val settled = arena.allocate(JAVA_INT, basicIndex.size.toLong())
            val deficiency = builder.invokeExact(handle, given, settled) as Int
            MemorySegment.copy(settled, JAVA_INT, 0L, repaired, 0, repaired.size)
            return deficiency
        }
    }

    fun snapshot(handle: MemorySegment): MemorySegment? {
        val taker = handlesOrThrow().snapshot ?: return null
        val taken = taker.invokeExact(handle) as MemorySegment
        return if (taken.address() == 0L) null else taken
    }

    fun restore(handle: MemorySegment, snapshot: MemorySegment): Boolean {
        val restorer = handlesOrThrow().restore ?: return false
        return restorer.invokeExact(handle, snapshot) as Int == 0
    }

    fun freeSnapshot(snapshot: MemorySegment) {
        handlesOrThrow().snapshotFree?.invokeExact(snapshot)
    }

    fun refactorizeReason(handle: MemorySegment): Int =
        handlesOrThrow().refactorizeReason?.let { it.invokeExact(handle) as Int } ?: 0

    fun kernel(handle: MemorySegment, out: IntArray): Boolean {
        val reader = handlesOrThrow().kernel ?: return false
        Arena.ofConfined().use { arena ->
            val dimension = arena.allocate(JAVA_INT)
            val entries = arena.allocate(JAVA_INT)
            reader.invokeExact(handle, dimension, entries)
            out[0] = dimension.get(JAVA_INT, 0L)
            out[1] = entries.get(JAVA_INT, 0L)
        }
        return true
    }

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
