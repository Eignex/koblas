package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.internal.host.FfmLibrary.Companion.intOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.pointerOf
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
        )
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

    fun solve(factor: KluFactor, rhs: DoubleArray, transpose: Boolean) {
        val h = handlesOrThrow()
        val solve = if (transpose) h.transposedSolve else h.solve
        val status = solve.invokeExact(
            factor.symbolicHolder.get(ADDRESS, 0),
            factor.numericHolder.get(ADDRESS, 0),
            factor.n,
            1,
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
