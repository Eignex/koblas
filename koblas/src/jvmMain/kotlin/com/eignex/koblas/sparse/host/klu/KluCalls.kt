package com.eignex.koblas.sparse.host.klu

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/** The public 32-bit-index ABI exported by KLU 2. */
internal class KluCalls(private val config: KluConfig) {
    private val linker: Linker? = runCatching { Linker.nativeLinker() }.getOrNull()
    private val critical = Linker.Option.critical(true)
    private val lookup: SymbolLookup? by lazy { openKlu() }
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

    val libraryPresent: Boolean get() = linker != null && lookup != null
    val available: Boolean get() = handles != null

    private fun openKlu(): SymbolLookup? {
        val paths = config.libraryPath?.let(::listOf) ?: KLU_SONAMES
        for (path in paths) {
            val opened =
                try {
                    SymbolLookup.libraryLookup(path, Arena.global())
                } catch (_: IllegalArgumentException) {
                    continue
                } catch (_: UnsatisfiedLinkError) {
                    continue
                }
            if (opened.find("klu_defaults").isPresent) return opened
        }
        return null
    }

    private class MissingSymbol : RuntimeException()

    private fun bindAll(): Handles? = try {
        val found = lookup ?: return null
        fun bind(
            name: String,
            result: java.lang.foreign.MemoryLayout,
            vararg args: java.lang.foreign.MemoryLayout,
        ): MethodHandle {
            val address = found.find(name).orElse(null) ?: throw MissingSymbol()
            return checkNotNull(linker).downcallHandle(address, FunctionDescriptor.of(result, *args), critical)
        }
        Handles(
            defaults = bind("klu_defaults", JAVA_INT, ADDRESS),
            analyze = bind("klu_analyze", ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS),
            factor = bind("klu_factor", ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
            refactor = bind("klu_refactor", JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
            rcond = bind("klu_rcond", JAVA_INT, ADDRESS, ADDRESS, ADDRESS),
            solve = bind("klu_solve", JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS),
            transposedSolve = bind("klu_tsolve", JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS),
            freeSymbolic = bind("klu_free_symbolic", JAVA_INT, ADDRESS, ADDRESS),
            freeNumeric = bind("klu_free_numeric", JAVA_INT, ADDRESS, ADDRESS),
        )
    } catch (_: MissingSymbol) {
        null
    }

    private fun handlesOrThrow(): Handles = checkNotNull(handles) { "KLU 2 is not available" }

    private fun MethodHandle.call(vararg arguments: Any?): Any? = invokeWithArguments(*arguments)

    fun factorize(a: Int, colPtr: IntArray, rowIdx: IntArray, values: DoubleArray, equilibrate: Boolean): KluFactor? {
        val h = handlesOrThrow()
        val arena = Arena.ofShared()
        val common = arena.allocate(KLU_COMMON_BYTES)
        val symbolicHolder = arena.allocate(ADDRESS)
        val numericHolder = arena.allocate(ADDRESS)
        try {
            check((h.defaults.call(common) as Int) != 0) { "klu_defaults failed" }
            applyConfig(common, equilibrate)
            val symbolic = h.analyze.call(
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
            val numeric = h.factor.call(
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
            check((h.rcond.call(symbolic, numeric, common) as Int) != 0) { "klu_rcond failed" }
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
        config.useBtf?.let { common.set(JAVA_INT, KLU_COMMON_BTF, it.asNativeBoolean()) }
        config.ordering?.let { common.set(JAVA_INT, KLU_COMMON_ORDERING, it.nativeValue) }
        config.haltIfSingular?.let { common.set(JAVA_INT, KLU_COMMON_HALT_IF_SINGULAR, it.asNativeBoolean()) }
        val scaling = if (equilibrate) config.equilibratedScaling.nativeValue else KLU_SCALE_NONE
        common.set(JAVA_INT, KLU_COMMON_SCALE, scaling)
    }

    fun solve(factor: KluFactor, rhs: DoubleArray, transpose: Boolean) {
        val h = handlesOrThrow()
        val solve = if (transpose) h.transposedSolve else h.solve
        val status = solve.call(
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
        val status = h.refactor.call(
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
                h.rcond.call(
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
            h.freeNumeric.call(factor.numericHolder, factor.common)
        }
        if (factor.symbolicHolder.get(
                ADDRESS,
                0,
            ).address() != 0L
        ) {
            h.freeSymbolic.call(factor.symbolicHolder, factor.common)
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

internal val KLU_SONAMES = listOf("libklu.so.2", "libklu.2.dylib")

private const val KLU_COMMON_BYTES = 160L
private const val KLU_NUMERIC_BYTES = 168L
private const val KLU_COMMON_TOL = 0L
private const val KLU_COMMON_MEMGROW = 8L
private const val KLU_COMMON_INITMEM_AMD = 16L
private const val KLU_COMMON_INITMEM = 24L
private const val KLU_COMMON_MAXWORK = 32L
private const val KLU_COMMON_BTF = 40L
private const val KLU_COMMON_ORDERING = 44L
private const val KLU_COMMON_SCALE = 48L
private const val KLU_COMMON_HALT_IF_SINGULAR = 72L
private const val KLU_COMMON_STATUS = 76L
private const val KLU_COMMON_RCOND = 112L
private const val KLU_NUMERIC_LNZ = 8L
private const val KLU_NUMERIC_UNZ = 12L
private const val KLU_SCALE_NONE = 0
private const val KLU_SCALE_MAX = 2
private const val KLU_SINGULAR = 1

private fun Boolean.asNativeBoolean(): Int = if (this) 1 else 0

private val KluOrdering.nativeValue: Int
    get() = when (this) {
        KluOrdering.AMD -> 0
        KluOrdering.COLAMD -> 1
    }

private val KluScaling.nativeValue: Int
    get() = when (this) {
        KluScaling.NONE -> KLU_SCALE_NONE
        KluScaling.SUM -> 1
        KluScaling.MAX -> KLU_SCALE_MAX
    }
