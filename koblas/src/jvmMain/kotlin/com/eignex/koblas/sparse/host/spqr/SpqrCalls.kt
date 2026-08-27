package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_COMMON_BYTES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_COMMON_PRINT
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_BYTES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_D
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_DTYPE
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_NCOL
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_NROW
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_NZMAX
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_X
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_XTYPE
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_Z
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DOUBLE
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_REAL
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
        FfmLibrary.open(candidates(SPQR_SONAMES, config.libraryPath), "SuiteSparseQR_C_factorize", "libspqr")
    }

    private val cholmod: FfmLibrary by lazy {
        FfmLibrary.open(candidates(CHOLMOD_SONAMES, null), "cholmod_start", "libcholmod")
    }

    @Volatile
    private var bindingFailure: String? = null

    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val start: MethodHandle,
        val factorize: MethodHandle,
        val explicitQr: MethodHandle,
        val qmult: MethodHandle,
        val solve: MethodHandle,
        val freeQr: MethodHandle,
        val freeDense: MethodHandle,
        val freeSparse: MethodHandle,
        val free: MethodHandle,
    )

    private val common: MemorySegment? by lazy { startCommon() }

    val available: Boolean get() = common != null

    val unavailableReason: String?
        get() = when {
            !spqr.present -> "libspqr could not be opened; SPQR does not appear to be installed"
            !cholmod.present -> "libspqr opened but libcholmod, which it runs its workspace against, did not"
            handles == null -> "libspqr opened but its symbols did not bind: ${bindingFailure ?: "unknown"}"
            common == null -> "libspqr opened but cholmod_start failed"
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
            factorize = spqr.handle(
                "SuiteSparseQR_C_factorize",
                FfmLibrary.pointerOf(JAVA_INT, JAVA_DOUBLE, ADDRESS, ADDRESS),
                critical = false,
            ),
            explicitQr = spqr.handle(
                "SuiteSparseQR_i_C_QR",
                FfmLibrary.intOf(JAVA_INT, JAVA_DOUBLE, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
                critical = false,
            ),
            qmult = spqr.handle(
                "SuiteSparseQR_C_qmult",
                FfmLibrary.pointerOf(JAVA_INT, ADDRESS, ADDRESS, ADDRESS),
                critical = false,
            ),
            solve = spqr.handle(
                "SuiteSparseQR_C_solve",
                FfmLibrary.pointerOf(JAVA_INT, ADDRESS, ADDRESS, ADDRESS),
                critical = false,
            ),
            freeQr = spqr.handle("SuiteSparseQR_C_free", FfmLibrary.intOf(ADDRESS, ADDRESS), critical = false),
            freeDense = cholmod.handle("cholmod_free_dense", FfmLibrary.intOf(ADDRESS, ADDRESS), critical = false),
            freeSparse = cholmod.handle(
                "cholmod_free_sparse",
                FfmLibrary.intOf(ADDRESS, ADDRESS),
                critical = false,
            ),
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

    private fun startCommon(): MemorySegment? {
        val bound = handles ?: return null
        // The global arena keeps the common alive for as long as any factor made against it.
        val block = Arena.global().allocate(CHOLMOD_COMMON_BYTES)
        if (bound.start.invokeExact(block) as Int != CHOLMOD_TRUE) return null
        block.set(JAVA_INT, CHOLMOD_COMMON_PRINT, 0)
        return block
    }

    fun factorize(a: CholmodMatrix): SpqrFactor? {
        val bound = handles ?: return null
        val shared = common ?: return null
        val factor = bound.factorize.invokeExact(
            config.options.ordering.code(),
            config.options.rankTolerance,
            a.segment,
            shared,
        ) as MemorySegment
        if (factor.address() == 0L) return null
        return SpqrFactor(factor, shared)
    }

    fun solveLeastSquares(factor: SpqrFactor, b: DoubleArray, cols: Int): DoubleArray? {
        val bound = handles ?: return null
        return Arena.ofConfined().use { arena ->
            val rhs = denseOf(arena, b)
            val projected = bound.qmult.invokeExact(SPQR_QTX, factor.segment, rhs, factor.common) as MemorySegment
            if (projected.address() == 0L) return@use null
            try {
                val solved = bound.solve.invokeExact(
                    SPQR_RETX_EQUALS_B,
                    factor.segment,
                    projected,
                    factor.common,
                ) as MemorySegment
                if (solved.address() == 0L) return@use null
                try {
                    read(solved, cols)
                } finally {
                    freeDense(arena, bound, solved, factor.common)
                }
            } finally {
                freeDense(arena, bound, projected, factor.common)
            }
        }
    }

    fun explicitR(a: CholmodMatrix, cols: Int): SpqrExplicit? {
        val bound = handles ?: return null
        val shared = common ?: return null
        return Arena.ofConfined().use { arena ->
            val rSlot = arena.allocate(ADDRESS)
            val eSlot = arena.allocate(ADDRESS)
            rSlot.set(ADDRESS, 0L, MemorySegment.NULL)
            eSlot.set(ADDRESS, 0L, MemorySegment.NULL)
            val rank = bound.explicitQr.invokeExact(
                config.options.ordering.code(),
                config.options.rankTolerance,
                cols,
                a.segment,
                MemorySegment.NULL,
                rSlot,
                eSlot,
                shared,
            ) as Int
            val r = rSlot.get(ADDRESS, 0L)
            if (rank < 0 || r.address() == 0L) return@use null
            try {
                val permutation = readPermutation(eSlot.get(ADDRESS, 0L), cols)
                SpqrExplicit(rank, readSparse(r), permutation)
            } finally {
                rSlot.set(ADDRESS, 0L, r)
                bound.freeSparse.invokeExact(rSlot, shared) as Int
                releasePermutation(bound, eSlot, cols, shared)
            }
        }
    }

    fun applyQ(factor: SpqrFactor, method: Int, x: DoubleArray, rows: Int): DoubleArray? {
        val bound = handles ?: return null
        return Arena.ofConfined().use { arena ->
            val operand = denseOf(arena, x)
            val applied = bound.qmult.invokeExact(method, factor.segment, operand, factor.common) as MemorySegment
            if (applied.address() == 0L) return@use null
            try {
                read(applied, rows)
            } finally {
                freeDense(arena, bound, applied, factor.common)
            }
        }
    }

    fun free(factor: SpqrFactor) {
        val bound = handles ?: return
        Arena.ofConfined().use { arena ->
            val slot = arena.allocate(ADDRESS)
            slot.set(ADDRESS, 0L, factor.segment)
            bound.freeQr.invokeExact(slot, factor.common) as Int
        }
    }

    private fun readPermutation(e: MemorySegment, cols: Int): IntArray {
        if (e.address() == 0L) return IntArray(cols) { it }
        val order = IntArray(cols)
        MemorySegment.copy(e.reinterpret(cols.toLong() * Int.SIZE_BYTES), JAVA_INT, 0L, order, 0, cols)
        return order
    }

    private fun releasePermutation(bound: Handles, slot: MemorySegment, cols: Int, common: MemorySegment) {
        val e = slot.get(ADDRESS, 0L)
        if (e.address() == 0L) return
        bound.free.invokeExact(cols.toLong(), Int.SIZE_BYTES.toLong(), e, common) as MemorySegment
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

    private fun read(dense: MemorySegment, cols: Int): DoubleArray {
        val descriptor = dense.reinterpret(CHOLMOD_DENSE_BYTES)
        val out = DoubleArray(cols)
        val values = descriptor.get(ADDRESS, CHOLMOD_DENSE_X).reinterpret(cols.toLong() * Double.SIZE_BYTES)
        MemorySegment.copy(values, JAVA_DOUBLE, 0L, out, 0, cols)
        return out
    }

    private fun freeDense(arena: Arena, bound: Handles, dense: MemorySegment, common: MemorySegment) {
        val slot = arena.allocate(ADDRESS)
        slot.set(ADDRESS, 0L, dense)
        bound.freeDense.invokeExact(slot, common) as Int
    }

    private fun denseOf(arena: Arena, values: DoubleArray): MemorySegment {
        val buffer = arena.allocate(JAVA_DOUBLE, maxOf(values.size, 1).toLong())
        MemorySegment.copy(values, 0, buffer, JAVA_DOUBLE, 0L, values.size)
        val dense = arena.allocate(CHOLMOD_DENSE_BYTES)
        dense.set(JAVA_LONG, CHOLMOD_DENSE_NROW, values.size.toLong())
        dense.set(JAVA_LONG, CHOLMOD_DENSE_NCOL, 1L)
        dense.set(JAVA_LONG, CHOLMOD_DENSE_NZMAX, values.size.toLong())
        dense.set(JAVA_LONG, CHOLMOD_DENSE_D, values.size.toLong())
        dense.set(ADDRESS, CHOLMOD_DENSE_X, buffer)
        dense.set(ADDRESS, CHOLMOD_DENSE_Z, MemorySegment.NULL)
        dense.set(JAVA_INT, CHOLMOD_DENSE_XTYPE, CHOLMOD_REAL)
        dense.set(JAVA_INT, CHOLMOD_DENSE_DTYPE, CHOLMOD_DOUBLE)
        return dense
    }
}

internal class SpqrFactor(val segment: MemorySegment, val common: MemorySegment)

internal class SpqrExplicit(val rank: Int, val r: F64SparseMatrix, val columnOrder: IntArray)
