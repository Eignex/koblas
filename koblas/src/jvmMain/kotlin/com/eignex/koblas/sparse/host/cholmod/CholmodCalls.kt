package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.FfmLibrary
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandle

/**
 * The 32-bit-index `cholmod_*` family, whose `int32_t` indices match IntArray; the `cholmod_l_*` family
 * takes `int64_t` and would need widening copies.
 */
internal class CholmodCalls(private val config: CholmodConfig) {

    /** Opened lazily, since a binding constructed during discovery must not load a library to exist. */
    private val library: FfmLibrary by lazy {
        FfmLibrary.open(candidates(), "cholmod_start", "libcholmod")
    }

    @Volatile
    private var bindingFailure: String? = null

    /** Bound lazily because `Linker.downcallHandle` is stack-hungry and discovery runs at arbitrary depth. */
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val start: MethodHandle,
        val finish: MethodHandle,
        val analyze: MethodHandle,
        val factorize: MethodHandle,
        val solve: MethodHandle,
        val rcond: MethodHandle,
        val sdmult: MethodHandle,
        val ssmult: MethodHandle,
        val freeFactor: MethodHandle,
        val freeDense: MethodHandle,
        val freeSparse: MethodHandle,
        val copyFactor: MethodHandle?,
        val changeFactor: MethodHandle?,
    )

    /**
     * The library's own `cholmod_common`, started once and shared by every factorization this binding makes.
     *
     * Shared rather than per factorization because starting one is what CHOLMOD charges for its workspace,
     * and because a factor has to be freed against the same one it was made with. It is never finished: the
     * factors outlive any scope this could close in, and the process exiting reclaims it.
     */
    private val common: MemorySegment? by lazy { startCommon(CHOLMOD_TRUE) }

    /**
     * A second common asking for `L·D·Lᵀ`, which is CHOLMOD's own default. Two rather than one flipped per
     * call, because the flag lives in the common and a factorization sharing it with another thread's would
     * get whichever kind that thread asked for.
     */
    private val commonLdl: MemorySegment? by lazy { startCommon(0) }

    /** Whether a libcholmod carrying the 32-bit family opened, which creates no downcall handle. */
    val libraryPresent: Boolean get() = library.present

    /** Whether the library opened, its symbols bound, and a common started. Reading this binds the handles. */
    val available: Boolean get() = common != null

    /** Why CHOLMOD is unusable, or null when it is usable. For diagnostics, not control flow. */
    val unavailableReason: String?
        get() = when {
            !library.present -> "libcholmod could not be opened; CHOLMOD does not appear to be installed"
            handles == null -> "libcholmod opened but its symbols did not bind: ${bindingFailure ?: "unknown"}"
            common == null -> "libcholmod opened but cholmod_start failed"
            else -> null
        }

    private fun candidates(): List<String> = buildList {
        config.libraryPath?.let(::add)
        config.searchDirectory?.let { directory ->
            for (soname in CHOLMOD_SONAMES) if ('/' !in soname) add("$directory/$soname")
        }
        addAll(CHOLMOD_SONAMES)
    }

    private fun bindAll(): Handles? = try {
        Handles(
            // start and finish allocate, which a critical downcall is not allowed to do.
            start = library.handle("cholmod_start", FfmLibrary.intOf(ADDRESS), critical = false),
            finish = library.handle("cholmod_finish", FfmLibrary.intOf(ADDRESS), critical = false),
            analyze = library.handle("cholmod_analyze", FfmLibrary.pointerOf(ADDRESS, ADDRESS), critical = false),
            factorize = library.handle(
                "cholmod_factorize",
                FfmLibrary.intOf(ADDRESS, ADDRESS, ADDRESS),
                critical = false,
            ),
            solve = library.handle(
                "cholmod_solve",
                FfmLibrary.pointerOf(JAVA_INT, ADDRESS, ADDRESS, ADDRESS),
                critical = false,
            ),
            rcond = library.handle("cholmod_rcond", FfmLibrary.doubleOf(ADDRESS, ADDRESS), critical = false),
            sdmult = library.handle(
                "cholmod_sdmult",
                FfmLibrary.intOf(ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
                critical = false,
            ),
            ssmult = library.handle(
                "cholmod_ssmult",
                FfmLibrary.pointerOf(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS),
                critical = false,
            ),
            freeFactor = library.handle("cholmod_free_factor", FfmLibrary.intOf(ADDRESS, ADDRESS), critical = false),
            freeDense = library.handle("cholmod_free_dense", FfmLibrary.intOf(ADDRESS, ADDRESS), critical = false),
            freeSparse = library.handle("cholmod_free_sparse", FfmLibrary.intOf(ADDRESS, ADDRESS), critical = false),
            // Optional: without them a factorization still solves and only reading its factors is refused.
            copyFactor = library.handleOrNull(
                "cholmod_copy_factor",
                FfmLibrary.pointerOf(ADDRESS, ADDRESS),
                critical = false,
            ),
            changeFactor = library.handleOrNull(
                "cholmod_change_factor",
                FfmLibrary.intOf(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS),
                critical = false,
            ),
        )
    } catch (failure: IllegalStateException) {
        bindingFailure = failure.message
        null
    }

    private fun startCommon(finalLl: Int): MemorySegment? {
        val bound = handles ?: return null
        // The global arena keeps the common alive for as long as any factor made against it.
        val block = Arena.global().allocate(CHOLMOD_COMMON_BYTES)
        if (bound.start.invokeExact(block) as Int != CHOLMOD_TRUE) return null
        block.set(JAVA_INT, CHOLMOD_COMMON_FINAL_LL, finalLl)
        block.set(JAVA_INT, CHOLMOD_COMMON_PRINT, 0)
        return block
    }

    /**
     * Analyze and factorize [a]'s lower triangle into `L·Lᵀ`, or into `L·D·Lᵀ` when [ldl], or null when the
     * library is unusable. The caller owns the returned factor and frees it with [free].
     */
    fun factorize(a: CholmodMatrix, ldl: Boolean = false): CholmodFactor? {
        val bound = handles ?: return null
        val shared = (if (ldl) commonLdl else common) ?: return null
        val factor = bound.analyze.invokeExact(a.segment, shared) as MemorySegment
        if (factor.address() == 0L) return null
        val reinterpreted = factor.reinterpret(CHOLMOD_FACTOR_BYTES)
        bound.factorize.invokeExact(a.segment, reinterpreted, shared) as Int
        return CholmodFactor(reinterpreted, shared)
    }

    /** Solves `A x = b` in place over [x], using caller-retained native [workspace]. */
    fun solve(factor: CholmodFactor, x: DoubleArray, workspace: CholmodSolveWorkspace): Boolean {
        val bound = handles ?: return false
        MemorySegment.copy(x, 0, workspace.rhs, JAVA_DOUBLE, 0L, x.size)
        val solved = bound.solve.invokeExact(
            CHOLMOD_A,
            factor.segment,
            workspace.dense,
            factor.common,
        ) as MemorySegment
        if (solved.address() == 0L) return false
        val answer = solved.reinterpret(CHOLMOD_DENSE_BYTES)
        val values = answer.get(ADDRESS, CHOLMOD_DENSE_X).reinterpret(x.size.toLong() * Double.SIZE_BYTES)
        MemorySegment.copy(values, JAVA_DOUBLE, 0L, x, 0, x.size)
        workspace.slot.set(ADDRESS, 0L, solved)
        bound.freeDense.invokeExact(workspace.slot, factor.common) as Int
        return true
    }

    /** Multiplies two prepared general sparse descriptors and copies the CSC result back. */
    fun ssmult(a: CholmodMatrix, b: CholmodMatrix): F64SparseMatrix? {
        val bound = handles ?: return null
        val shared = common ?: return null
        val product = bound.ssmult.invokeExact(
            a.segment,
            b.segment,
            CHOLMOD_STYPE_GENERAL,
            CHOLMOD_TRUE,
            CHOLMOD_TRUE,
            shared,
        ) as MemorySegment
        if (product.address() == 0L) return null
        return Arena.ofConfined().use { arena ->
            try {
                val descriptor = product.reinterpret(CHOLMOD_SPARSE_BYTES)
                val rows = descriptor.get(JAVA_LONG, CHOLMOD_SPARSE_NROW).toInt()
                val cols = descriptor.get(JAVA_LONG, CHOLMOD_SPARSE_NCOL).toInt()
                val colPtr = IntArray(cols + 1)
                val p = descriptor.get(ADDRESS, CHOLMOD_SPARSE_P).reinterpret((cols + 1L) * Int.SIZE_BYTES)
                MemorySegment.copy(p, JAVA_INT, 0L, colPtr, 0, colPtr.size)
                val nnz = colPtr[cols]
                val rowIdx = IntArray(nnz)
                val values = DoubleArray(nnz)
                val i = descriptor.get(ADDRESS, CHOLMOD_SPARSE_I).reinterpret(nnz.toLong() * Int.SIZE_BYTES)
                val x = descriptor.get(ADDRESS, CHOLMOD_SPARSE_X).reinterpret(nnz.toLong() * Double.SIZE_BYTES)
                MemorySegment.copy(i, JAVA_INT, 0L, rowIdx, 0, nnz)
                MemorySegment.copy(x, JAVA_DOUBLE, 0L, values, 0, nnz)
                F64SparseMatrix.wrap(rows, cols, colPtr, rowIdx, values)
            } finally {
                val slot = arena.allocate(ADDRESS)
                slot.set(ADDRESS, 0L, product)
                bound.freeSparse.invokeExact(slot, shared) as Int
            }
        }
    }

    /**
     * `L` and the fill-reducing permutation from [factor], or null when this libcholmod lacks the conversion
     * symbols.
     *
     * The factor is copied before conversion, because making it simplicial and packed rewrites it in place
     * and the original still has solves to answer. Once packed, its `p`, `i` and `x` are `L` in CSC.
     *
     * [asLl] picks which factorization to read: `L·Lᵀ` keeps the real diagonal, and `L·D·Lᵀ` puts `D` on the
     * diagonal of `L`, which the caller splits out.
     */
    fun extractFactor(factor: CholmodFactor, asLl: Boolean): CholmodFactors? {
        val bound = handles ?: return null
        val copy = bound.copyFactor ?: return null
        val change = bound.changeFactor ?: return null
        val duplicate = copy.invokeExact(factor.segment, factor.common) as MemorySegment
        if (duplicate.address() == 0L) return null
        try {
            return readFactor(change, duplicate, factor.common, asLl)
        } finally {
            Arena.ofConfined().use { arena ->
                val slot = arena.allocate(ADDRESS)
                slot.set(ADDRESS, 0L, duplicate)
                bound.freeFactor.invokeExact(slot, factor.common) as Int
            }
        }
    }

    private fun readFactor(
        change: MethodHandle,
        duplicate: MemorySegment,
        common: MemorySegment,
        asLl: Boolean,
    ): CholmodFactors? {
        val block = duplicate.reinterpret(CHOLMOD_FACTOR_BYTES)
        val converted = change.invokeExact(
            CHOLMOD_REAL_XTYPE,
            if (asLl) CHOLMOD_TRUE else 0,
            0,
            CHOLMOD_TRUE,
            CHOLMOD_TRUE,
            block,
            common,
        ) as Int
        if (converted != CHOLMOD_TRUE) return null
        val order = block.get(JAVA_LONG, CHOLMOD_FACTOR_N).toInt()
        val colPtr = IntArray(order + 1)
        MemorySegment.copy(
            block.get(ADDRESS, CHOLMOD_FACTOR_P).reinterpret((order + 1L) * Int.SIZE_BYTES),
            JAVA_INT,
            0L,
            colPtr,
            0,
            colPtr.size,
        )
        val nonzeros = colPtr[order]
        val rowIdx = IntArray(nonzeros)
        val values = DoubleArray(nonzeros)
        if (nonzeros > 0) {
            val indices = block.get(ADDRESS, CHOLMOD_FACTOR_I).reinterpret(nonzeros.toLong() * Int.SIZE_BYTES)
            val entries = block.get(ADDRESS, CHOLMOD_FACTOR_X).reinterpret(nonzeros.toLong() * Double.SIZE_BYTES)
            MemorySegment.copy(indices, JAVA_INT, 0L, rowIdx, 0, nonzeros)
            MemorySegment.copy(entries, JAVA_DOUBLE, 0L, values, 0, nonzeros)
        }
        return CholmodFactors(order, colPtr, rowIdx, values, permutation(block, order))
    }

    /** The fill-reducing ordering, or the identity where CHOLMOD did not reorder. */
    private fun permutation(block: MemorySegment, order: Int): IntArray {
        val perm = block.get(ADDRESS, CHOLMOD_FACTOR_PERM)
        if (perm.address() == 0L) return IntArray(order) { it }
        val out = IntArray(order)
        MemorySegment.copy(perm.reinterpret(order.toLong() * Int.SIZE_BYTES), JAVA_INT, 0L, out, 0, order)
        return out
    }

    /** CHOLMOD's own reciprocal condition estimate for [factor]. */
    fun rcond(factor: CholmodFactor): Double {
        val bound = handles ?: return 0.0
        return bound.rcond.invokeExact(factor.segment, factor.common) as Double
    }

    /**
     * `y = alpha · op(A) · x + beta · y` over [columns] right-hand sides held column-major in [x] and [y].
     *
     * The scalars are two-element arrays because CHOLMOD takes them as complex pairs; only the first is read
     * for a real matrix.
     */
    @Suppress("LongParameterList") // the routine's own signature, plus the shape the two dense blocks share
    fun sdmult(
        a: CholmodMatrix,
        transpose: Boolean,
        alpha: Double,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        columns: Int,
        xRows: Int,
        yRows: Int,
    ): Boolean {
        val bound = handles ?: return false
        val shared = common ?: return false
        Arena.ofConfined().use { arena ->
            val alphaPair = arena.allocate(JAVA_DOUBLE, 2L).also { it.setAtIndex(JAVA_DOUBLE, 0L, alpha) }
            val betaPair = arena.allocate(JAVA_DOUBLE, 2L).also { it.setAtIndex(JAVA_DOUBLE, 0L, beta) }
            val xBlock = arena.allocate(JAVA_DOUBLE, x.size.toLong())
            MemorySegment.copy(x, 0, xBlock, JAVA_DOUBLE, 0L, x.size)
            val yBlock = arena.allocate(JAVA_DOUBLE, y.size.toLong())
            MemorySegment.copy(y, 0, yBlock, JAVA_DOUBLE, 0L, y.size)
            val dense = { rows: Int, block: MemorySegment ->
                arena.allocate(CHOLMOD_DENSE_BYTES).also {
                    it.set(JAVA_LONG, CHOLMOD_DENSE_NROW, rows.toLong())
                    it.set(JAVA_LONG, CHOLMOD_DENSE_NCOL, columns.toLong())
                    it.set(JAVA_LONG, CHOLMOD_DENSE_NZMAX, rows.toLong() * columns)
                    it.set(JAVA_LONG, CHOLMOD_DENSE_D, rows.toLong())
                    it.set(ADDRESS, CHOLMOD_DENSE_X, block)
                    it.set(ADDRESS, CHOLMOD_DENSE_Z, MemorySegment.NULL)
                    it.set(JAVA_INT, CHOLMOD_DENSE_XTYPE, CHOLMOD_REAL)
                    it.set(JAVA_INT, CHOLMOD_DENSE_DTYPE, CHOLMOD_DOUBLE)
                }
            }
            val xDense = dense(xRows, xBlock)
            val yDense = dense(yRows, yBlock)
            val flag = if (transpose) CHOLMOD_TRUE else 0
            val ok = bound.sdmult.invokeExact(
                a.segment,
                flag,
                alphaPair,
                betaPair,
                xDense,
                yDense,
                shared,
            ) as Int
            if (ok != CHOLMOD_TRUE) return false
            MemorySegment.copy(yBlock, JAVA_DOUBLE, 0L, y, 0, y.size)
        }
        return true
    }

    /** Frees the native factor. */
    fun free(factor: CholmodFactor) {
        val bound = handles ?: return
        Arena.ofConfined().use { arena ->
            val slot = arena.allocate(ADDRESS)
            slot.set(ADDRESS, 0L, factor.segment)
            bound.freeFactor.invokeExact(slot, factor.common) as Int
        }
    }
}

/** Native input descriptor retained for repeated solves against one CHOLMOD factor. */
internal class CholmodSolveWorkspace(n: Int, columns: Int = 1) : AutoCloseable {
    private val arena = Arena.ofShared()
    val rhs: MemorySegment = arena.allocate(JAVA_DOUBLE, maxOf(n.toLong() * columns, 1L))
    val dense: MemorySegment = arena.allocate(CHOLMOD_DENSE_BYTES)
    val slot: MemorySegment = arena.allocate(ADDRESS)

    init {
        dense.set(JAVA_LONG, CHOLMOD_DENSE_NROW, n.toLong())
        dense.set(JAVA_LONG, CHOLMOD_DENSE_NCOL, columns.toLong())
        dense.set(JAVA_LONG, CHOLMOD_DENSE_NZMAX, n.toLong() * columns)
        dense.set(JAVA_LONG, CHOLMOD_DENSE_D, n.toLong())
        dense.set(ADDRESS, CHOLMOD_DENSE_X, rhs)
        dense.set(ADDRESS, CHOLMOD_DENSE_Z, MemorySegment.NULL)
        dense.set(JAVA_INT, CHOLMOD_DENSE_XTYPE, CHOLMOD_REAL)
        dense.set(JAVA_INT, CHOLMOD_DENSE_DTYPE, CHOLMOD_DOUBLE)
    }

    override fun close(): Unit = arena.close()
}

/** One native factor, and the common it was made against and must be freed against. */
internal class CholmodFactor(val segment: MemorySegment, val common: MemorySegment) {
    val n: Int get() = segment.get(JAVA_LONG, CHOLMOD_FACTOR_N).toInt()

    /** The column whose pivot was not positive, or [n] when the factorization succeeded. */
    val minor: Int get() = segment.get(JAVA_LONG, CHOLMOD_FACTOR_MINOR).toInt()

    val nzmax: Int get() = segment.get(JAVA_LONG, CHOLMOD_FACTOR_NZMAX).toInt()

    /** Whether the factor is `L·Lᵀ` rather than `L·D·Lᵀ`, which is what CHOLMOD squares its estimate for. */
    val isLl: Boolean get() = segment.get(JAVA_INT, CHOLMOD_FACTOR_IS_LL) == CHOLMOD_TRUE
}
