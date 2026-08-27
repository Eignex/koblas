package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.internal.host.FfmLibrary.Companion.intOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.pointerOf
import com.eignex.koblas.sparse.internal.transposeRaw
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

/** The public 32-bit-index ABI exported by KLU 2. */
internal class KluCalls(private val config: KluConfig) {
    private val library: FfmLibrary by lazy {
        FfmLibrary.open(config.libraryPath?.let(::listOf) ?: KLU_SONAMES, "klu_defaults", "KLU 2")
    }
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val defaults: MethodHandle,
        val analyze: MethodHandle,
        val factor: MethodHandle,
        val refactor: MethodHandle,
        val rcond: MethodHandle,
        val solve: MethodHandle,
        val transposedSolve: MethodHandle,
        val freeSymbolic: MethodHandle,
        val freeNumeric: MethodHandle,
        val extract: MethodHandle?,
    )

    val libraryPresent: Boolean get() = library.present
    val available: Boolean get() = handles != null

    private fun bindAll(): Handles? {
        if (!library.present) return null
        return Handles(
            defaults = library.handleOrNull("klu_defaults", intOf(ADDRESS)) ?: return null,
            analyze = library.handleOrNull("klu_analyze", pointerOf(JAVA_INT, ADDRESS, ADDRESS, ADDRESS))
                ?: return null,
            factor = library.handleOrNull("klu_factor", pointerOf(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS))
                ?: return null,
            refactor = library.handleOrNull(
                "klu_refactor",
                intOf(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
            ) ?: return null,
            rcond = library.handleOrNull("klu_rcond", intOf(ADDRESS, ADDRESS, ADDRESS)) ?: return null,
            solve = library.handleOrNull("klu_solve", intOf(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS))
                ?: return null,
            transposedSolve = library.handleOrNull(
                "klu_tsolve",
                intOf(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS),
            ) ?: return null,
            freeSymbolic = library.handleOrNull("klu_free_symbolic", intOf(ADDRESS, ADDRESS)) ?: return null,
            freeNumeric = library.handleOrNull("klu_free_numeric", intOf(ADDRESS, ADDRESS)) ?: return null,
            // Optional, unlike the rest: without it a factorization still solves and only reading its
            // factors is refused.
            extract = library.handleOrNull("klu_extract", intOf(*Array(16) { ADDRESS })),
        )
    }

    /** [extractFactors]'s implementation, which needs the bound handles. */
    internal fun extract(factor: KluFactor): KluFactors? {
        val bound = handles ?: return null
        val extract = bound.extract ?: return null
        val numeric = factor.numeric
        val order = factor.n
        val lNonzeros = numeric.get(JAVA_INT, KLU_NUMERIC_LNZ)
        val uNonzeros = numeric.get(JAVA_INT, KLU_NUMERIC_UNZ)
        val offNonzeros = numeric.get(JAVA_INT, KLU_NUMERIC_NZOFF)
        val blocks = numeric.get(JAVA_INT, KLU_NUMERIC_NBLOCKS)
        return Arena.ofConfined().use { arena ->
            val lPtr = arena.allocate(JAVA_INT, order + 1L)
            val lIdx = arena.allocate(JAVA_INT, maxOf(lNonzeros, 1).toLong())
            val lVal = arena.allocate(JAVA_DOUBLE, maxOf(lNonzeros, 1).toLong())
            val uPtr = arena.allocate(JAVA_INT, order + 1L)
            val uIdx = arena.allocate(JAVA_INT, maxOf(uNonzeros, 1).toLong())
            val uVal = arena.allocate(JAVA_DOUBLE, maxOf(uNonzeros, 1).toLong())
            val fPtr = arena.allocate(JAVA_INT, order + 1L)
            val fIdx = arena.allocate(JAVA_INT, maxOf(offNonzeros, 1).toLong())
            val fVal = arena.allocate(JAVA_DOUBLE, maxOf(offNonzeros, 1).toLong())
            val rowPerm = arena.allocate(JAVA_INT, order.toLong())
            val colPerm = arena.allocate(JAVA_INT, order.toLong())
            val scaling = arena.allocate(JAVA_DOUBLE, order.toLong())
            val boundaries = arena.allocate(JAVA_INT, blocks + 1L)
            val ok = extract.invokeExact(
                numeric, factor.symbolicHolder.get(ADDRESS, 0),
                lPtr, lIdx, lVal, uPtr, uIdx, uVal, fPtr, fIdx, fVal,
                rowPerm, colPerm, scaling, boundaries, factor.common,
            ) as Int
            if (ok != 1) return@use null
            KluFactors(
                lower = matrix(order, lPtr, lIdx, lVal, lNonzeros),
                upper = matrix(order, uPtr, uIdx, uVal, uNonzeros),
                offDiagonal = matrix(order, fPtr, fIdx, fVal, offNonzeros),
                rowOrder = readInts(rowPerm, order),
                columnOrder = readInts(colPerm, order),
                rowScaling = readDoubles(scaling, order).let { scale ->
                    // KLU divides rows by these, so the multiplier the interface documents is the reciprocal.
                    DoubleArray(order) { if (scale[it] == 0.0) 1.0 else 1.0 / scale[it] }
                },
            )
        }
    }

    /** A CSC matrix over copies of these segments, rows sorted by the transpose pair. */
    private fun matrix(
        order: Int,
        pointers: MemorySegment,
        indices: MemorySegment,
        values: MemorySegment,
        nonzeros: Int,
    ): F64SparseMatrix {
        val ptr = readInts(pointers, order + 1)
        val idx = readInts(indices, nonzeros)
        val entries = readDoubles(values, nonzeros)
        val once = transposeRaw(order, order, ptr, idx, entries)
        return transposeRaw(order, order, once.colPtr, once.rowIdx, once.values)
    }

    private fun readInts(segment: MemorySegment, count: Int): IntArray {
        val out = IntArray(count)
        if (count > 0) MemorySegment.copy(segment, JAVA_INT, 0L, out, 0, count)
        return out
    }

    private fun readDoubles(segment: MemorySegment, count: Int): DoubleArray {
        val out = DoubleArray(count)
        if (count > 0) MemorySegment.copy(segment, JAVA_DOUBLE, 0L, out, 0, count)
        return out
    }

    private fun handlesOrThrow(): Handles = checkNotNull(handles) { "KLU 2 is not available" }

    fun factorize(a: Int, colPtr: IntArray, rowIdx: IntArray, values: DoubleArray, equilibrate: Boolean): KluFactor? {
        val h = handlesOrThrow()
        val arena = Arena.ofShared()
        val common = arena.allocate(KLU_COMMON_BYTES)
        val symbolicHolder = arena.allocate(ADDRESS)
        val numericHolder = arena.allocate(ADDRESS)
        try {
            check((h.defaults.invokeExact(common) as Int) != 0) { "klu_defaults failed" }
            applyConfig(common, equilibrate)
            val symbolic = h.analyze.invokeExact(
                a,
                MemorySegment.ofArray(colPtr),
                MemorySegment.ofArray(rowIdx),
                common,
            ) as MemorySegment
            if (symbolic.address() == 0L) {
                error(
                    "klu_analyze failed with status ${common.get(JAVA_INT, KLU_COMMON_STATUS)}",
                )
            }
            symbolicHolder.set(ADDRESS, 0, symbolic)
            val numeric = h.factor.invokeExact(
                MemorySegment.ofArray(colPtr),
                MemorySegment.ofArray(rowIdx),
                MemorySegment.ofArray(values),
                symbolic,
                common,
            ) as MemorySegment
            if (numeric.address() == 0L) {
                if (common.get(JAVA_INT, KLU_COMMON_STATUS) == KLU_SINGULAR) return null
                error("klu_factor failed with status ${common.get(JAVA_INT, KLU_COMMON_STATUS)}")
            }
            numericHolder.set(ADDRESS, 0, numeric)
            check((h.rcond.invokeExact(symbolic, numeric, common) as Int) != 0) { "klu_rcond failed" }
            return KluFactor(
                a,
                arena,
                common,
                symbolicHolder,
                numericHolder,
                common.get(JAVA_DOUBLE, KLU_COMMON_RCOND),
                colPtr.copyOf(),
                rowIdx.copyOf(),
            )
        } catch (@Suppress("TooGenericExceptionCaught") t: Throwable) {
            free(KluFactor(a, arena, common, symbolicHolder, numericHolder, 0.0, colPtr.copyOf(), rowIdx.copyOf()))
            throw t
        }
    }

    private fun applyConfig(common: MemorySegment, equilibrate: Boolean) {
        config.pivotTolerance?.let { common.set(JAVA_DOUBLE, KLU_COMMON_TOL, it) }
        config.memoryGrowth?.let { common.set(JAVA_DOUBLE, KLU_COMMON_MEMGROW, it) }
        config.amdInitialMemoryFactor?.let { common.set(JAVA_DOUBLE, KLU_COMMON_INITMEM_AMD, it) }
        config.initialMemoryFactor?.let { common.set(JAVA_DOUBLE, KLU_COMMON_INITMEM, it) }
        config.maxBtfWork?.let { common.set(JAVA_DOUBLE, KLU_COMMON_MAXWORK, it) }
        config.useBtf?.let { common.set(JAVA_INT, KLU_COMMON_BTF, it.asNativeKluBoolean()) }
        config.ordering?.let { common.set(JAVA_INT, KLU_COMMON_ORDERING, it.nativeValue) }
        config.haltIfSingular?.let { common.set(JAVA_INT, KLU_COMMON_HALT_IF_SINGULAR, it.asNativeKluBoolean()) }
        val scaling = if (equilibrate) config.equilibratedScaling.nativeValue else KLU_SCALE_NONE
        common.set(JAVA_INT, KLU_COMMON_SCALE, scaling)
    }

    fun solve(factor: KluFactor, rhs: DoubleArray, transpose: Boolean, rightHandSides: Int = 1) {
        val h = handlesOrThrow()
        val solve = if (transpose) h.transposedSolve else h.solve
        val status = solve.invokeExact(
            factor.symbolicHolder.get(ADDRESS, 0),
            factor.numericHolder.get(ADDRESS, 0),
            factor.n,
            rightHandSides,
            MemorySegment.ofArray(rhs),
            factor.common,
        ) as Int
        check(status != 0) { "KLU solve failed with status ${factor.common.get(JAVA_INT, KLU_COMMON_STATUS)}" }
    }

    fun refactor(
        factor: KluFactor,
        colPtr: IntArray,
        rowIdx: IntArray,
        values: DoubleArray,
        equilibrate: Boolean,
    ): Boolean {
        val h = handlesOrThrow()
        applyConfig(factor.common, equilibrate)
        val status = h.refactor.invokeExact(
            MemorySegment.ofArray(colPtr),
            MemorySegment.ofArray(rowIdx),
            MemorySegment.ofArray(values),
            factor.symbolicHolder.get(ADDRESS, 0),
            factor.numericHolder.get(ADDRESS, 0),
            factor.common,
        ) as Int
        if (factor.common.get(JAVA_INT, KLU_COMMON_STATUS) == KLU_SINGULAR) return false
        check(status != 0) { "klu_refactor failed with status ${factor.common.get(JAVA_INT, KLU_COMMON_STATUS)}" }
        check(
            (
                h.rcond.invokeExact(
                    factor.symbolicHolder.get(ADDRESS, 0),
                    factor.numericHolder.get(ADDRESS, 0),
                    factor.common,
                ) as Int
                ) != 0,
        ) { "klu_rcond failed" }
        factor.rcond = factor.common.get(JAVA_DOUBLE, KLU_COMMON_RCOND)
        return true
    }

    fun free(factor: KluFactor) {
        val h = handles ?: return
        if (factor.numericHolder.get(
                ADDRESS,
                0,
            ).address() != 0L
        ) {
            // The cast is the descriptor, not a use of the status: klu_free_numeric returns int, and
            // without it Kotlin infers Unit here and emits a void call the handle will not accept.
            h.freeNumeric.invokeExact(factor.numericHolder, factor.common) as Int
        }
        if (factor.symbolicHolder.get(
                ADDRESS,
                0,
            ).address() != 0L
        ) {
            h.freeSymbolic.invokeExact(factor.symbolicHolder, factor.common) as Int // as above
        }
    }
}

/**
 * `L`, `U`, the off-diagonal blocks, the two permutations and the scaling from [factor], or null when this
 * libklu lacks `klu_extract`.
 *
 * KLU permutes to block triangular form and factors the blocks, so the entries outside them are in neither
 * `L` nor `U` and come back as `F`. Its scaling divides, so it is inverted here to the multiplier the
 * factor interface documents.
 */
internal fun KluCalls.extractFactors(factor: KluFactor): KluFactors? = extract(factor)

internal class KluFactor(
    val n: Int,
    val arena: Arena,
    val common: MemorySegment,
    val symbolicHolder: MemorySegment,
    val numericHolder: MemorySegment,
    var rcond: Double,
    val colPtr: IntArray,
    val rowIdx: IntArray,
) {
    val numeric: MemorySegment get() = numericHolder.get(ADDRESS, 0).reinterpret(KLU_NUMERIC_BYTES)

    val nnz: Int
        get() {
            return numeric.get(JAVA_INT, KLU_NUMERIC_LNZ) + numeric.get(JAVA_INT, KLU_NUMERIC_UNZ)
        }
}
