package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_COMMON_BYTES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_COMMON_PRINT
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_BYTES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_NCOL
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_X
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SONAMES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_BYTES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_I
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_NCOL
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_NROW
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_P
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_X
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_TRUE
import com.eignex.koblas.sparse.host.cholmod.CholmodMatrix
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandle

internal class SpqrCalls(private val config: SpqrConfig) {
    private val spqr: FfmLibrary by lazy {
        FfmLibrary.open(candidates(SPQR_SONAMES, config.libraryPath), "SuiteSparseQR_i_C", "libspqr")
    }

    private val cholmod: FfmLibrary by lazy {
        FfmLibrary.open(candidates(CHOLMOD_SONAMES, null), "cholmod_start", "libcholmod")
    }

    @Volatile
    private var bindingFailure: String? = null

    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val start: MethodHandle,
        val finish: MethodHandle,
        val qr: MethodHandle,
        val freeDense: MethodHandle,
        val freeSparse: MethodHandle,
        val free: MethodHandle,
    )

    val available: Boolean get() = handles != null

    val unavailableReason: String?
        get() = when {
            !spqr.present -> "libspqr could not be opened; SPQR does not appear to be installed"
            !cholmod.present -> "libspqr opened but libcholmod, which it runs its workspace against, did not"
            handles == null -> "libspqr opened but its symbols did not bind: ${bindingFailure ?: "unknown"}"
            else -> null
        }

    private fun candidates(sonames: List<String>, explicit: String?): List<String> = buildList {
        explicit?.let(::add)
        config.searchDirectory?.let { directory ->
            for (soname in sonames) if ('/' !in soname) add("$directory/$soname")
        }
        addAll(sonames)
    }

    private fun bindAll(): Handles? = try {
        Handles(
            start = cholmod.handle("cholmod_start", FfmLibrary.intOf(ADDRESS), critical = false),
            finish = cholmod.handle("cholmod_finish", FfmLibrary.intOf(ADDRESS), critical = false),
            qr = spqr.handle(
                "SuiteSparseQR_i_C",
                FfmLibrary.intOf(
                    JAVA_INT,
                    JAVA_DOUBLE,
                    JAVA_INT,
                    JAVA_INT,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS,
                ),
                critical = false,
            ),
            freeDense = cholmod.handle("cholmod_free_dense", FfmLibrary.intOf(ADDRESS, ADDRESS), critical = false),
            freeSparse = cholmod.handle("cholmod_free_sparse", FfmLibrary.intOf(ADDRESS, ADDRESS), critical = false),
            free = cholmod.handle(
                "cholmod_free",
                FfmLibrary.pointerOf(JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS),
                critical = false,
            ),
        )
    } catch (failure: IllegalStateException) {
        bindingFailure = failure.message
        null
    }

    fun factorize(a: CholmodMatrix, rows: Int, cols: Int): SpqrFactorData? {
        val bound = handles ?: return null
        return Arena.ofConfined().use { arena ->
            val common = arena.allocate(CHOLMOD_COMMON_BYTES)
            if (bound.start.invokeExact(common) as Int != CHOLMOD_TRUE) return@use null
            common.set(JAVA_INT, CHOLMOD_COMMON_PRINT, 0)
            try {
                val rSlot = pointerSlot(arena)
                val eSlot = pointerSlot(arena)
                val hSlot = pointerSlot(arena)
                val hpSlot = pointerSlot(arena)
                val tauSlot = pointerSlot(arena)
                val rank = bound.qr.invokeExact(
                    config.options.ordering.code(),
                    config.options.rankTolerance,
                    cols,
                    0,
                    a.segment,
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    rSlot,
                    eSlot,
                    hSlot,
                    hpSlot,
                    tauSlot,
                    common,
                ) as Int
                try {
                    val r = rSlot.get(ADDRESS, 0L)
                    val h = hSlot.get(ADDRESS, 0L)
                    val tau = tauSlot.get(ADDRESS, 0L)
                    if (rank < 0 || r.address() == 0L || h.address() == 0L || tau.address() == 0L) return@use null
                    SpqrFactorData(
                        rank,
                        readSparse(r),
                        readPermutation(eSlot.get(ADDRESS, 0L), cols),
                        readSparse(h),
                        readPermutation(hpSlot.get(ADDRESS, 0L), rows),
                        readDense(tau),
                    )
                } finally {
                    freeSparse(bound, rSlot, common)
                    freePermutation(bound, eSlot, cols, common)
                    freeSparse(bound, hSlot, common)
                    freePermutation(bound, hpSlot, rows, common)
                    freeDense(bound, tauSlot, common)
                }
            } finally {
                bound.finish.invokeExact(common) as Int
            }
        }
    }

    private fun pointerSlot(arena: Arena): MemorySegment = arena.allocate(ADDRESS).also {
        it.set(ADDRESS, 0L, MemorySegment.NULL)
    }

    private fun readPermutation(pointer: MemorySegment, size: Int): IntArray {
        if (pointer.address() == 0L) return IntArray(size) { it }
        val out = IntArray(size)
        MemorySegment.copy(pointer.reinterpret(size.toLong() * Int.SIZE_BYTES), JAVA_INT, 0L, out, 0, size)
        return out
    }

    private fun readSparse(sparse: MemorySegment): F64SparseMatrix {
        val descriptor = sparse.reinterpret(CHOLMOD_SPARSE_BYTES)
        val rows = descriptor.get(JAVA_LONG, CHOLMOD_SPARSE_NROW).toInt()
        val cols = descriptor.get(JAVA_LONG, CHOLMOD_SPARSE_NCOL).toInt()
        val colPtr = IntArray(cols + 1)
        val p = descriptor.get(ADDRESS, CHOLMOD_SPARSE_P).reinterpret((cols + 1L) * Int.SIZE_BYTES)
        MemorySegment.copy(p, JAVA_INT, 0L, colPtr, 0, colPtr.size)
        val nnz = colPtr[cols]
        val rowIdx = IntArray(nnz)
        val values = DoubleArray(nnz)
        if (nnz > 0) {
            val i = descriptor.get(ADDRESS, CHOLMOD_SPARSE_I).reinterpret(nnz.toLong() * Int.SIZE_BYTES)
            val x = descriptor.get(ADDRESS, CHOLMOD_SPARSE_X).reinterpret(nnz.toLong() * Double.SIZE_BYTES)
            MemorySegment.copy(i, JAVA_INT, 0L, rowIdx, 0, nnz)
            MemorySegment.copy(x, JAVA_DOUBLE, 0L, values, 0, nnz)
        }
        return F64SparseMatrix.wrap(rows, cols, colPtr, rowIdx, values)
    }

    private fun readDense(dense: MemorySegment): DoubleArray {
        val descriptor = dense.reinterpret(CHOLMOD_DENSE_BYTES)
        val size = descriptor.get(JAVA_LONG, CHOLMOD_DENSE_NCOL).toInt()
        val out = DoubleArray(size)
        if (size > 0) {
            val values = descriptor.get(ADDRESS, CHOLMOD_DENSE_X).reinterpret(size.toLong() * Double.SIZE_BYTES)
            MemorySegment.copy(values, JAVA_DOUBLE, 0L, out, 0, size)
        }
        return out
    }

    private fun freeSparse(bound: Handles, slot: MemorySegment, common: MemorySegment) {
        if (slot.get(ADDRESS, 0L).address() != 0L) bound.freeSparse.invokeExact(slot, common) as Int
    }

    private fun freeDense(bound: Handles, slot: MemorySegment, common: MemorySegment) {
        if (slot.get(ADDRESS, 0L).address() != 0L) bound.freeDense.invokeExact(slot, common) as Int
    }

    private fun freePermutation(bound: Handles, slot: MemorySegment, size: Int, common: MemorySegment) {
        val pointer = slot.get(ADDRESS, 0L)
        if (pointer.address() != 0L) {
            bound.free.invokeExact(size.toLong(), Int.SIZE_BYTES.toLong(), pointer, common) as MemorySegment
        }
    }
}
