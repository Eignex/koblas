package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.internal.backend.nativeLibraryPaths
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
internal object KluCalls {
    private val linker: Linker? = runCatching { Linker.nativeLinker() }.getOrNull()
    private val critical = Linker.Option.critical(true)
    private val lookup: SymbolLookup? by lazy { openKlu() }
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val defaults: MethodHandle,
        val analyze: MethodHandle,
        val factor: MethodHandle,
        val solve: MethodHandle,
        val transposedSolve: MethodHandle,
        val freeSymbolic: MethodHandle,
        val freeNumeric: MethodHandle,
    )

    val libraryPresent: Boolean get() = linker != null && lookup != null
    val available: Boolean get() = handles != null

    private fun openKlu(): SymbolLookup? {
        val paths = nativeLibraryPaths("koblas.klu.path", "KOBLAS_KLU_PATH", KLU_SONAMES)
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
            common.set(JAVA_INT, KLU_COMMON_SCALE, if (equilibrate) KLU_SCALE_MAX else KLU_SCALE_NONE)
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
            return KluFactor(a, arena, common, symbolicHolder, numericHolder)
            // Native calls can fail after allocating factors.
        } catch (@Suppress("TooGenericExceptionCaught") t: Throwable) {
            free(KluFactor(a, arena, common, symbolicHolder, numericHolder))
            throw t
        }
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

    fun determinant(factor: KluFactor): Double {
        val numeric = factor.numeric
        val symbolic = factor.symbolic
        val diagonal = numeric.get(ADDRESS, KLU_NUMERIC_UDIAG).reinterpret(factor.n.toLong() * JAVA_DOUBLE.byteSize())
        var determinant = 1.0
        for (i in 0 until factor.n) determinant *= diagonal.getAtIndex(JAVA_DOUBLE, i.toLong())
        determinant *= permutationSign(numeric.get(ADDRESS, KLU_NUMERIC_PNUM), factor.n)
        determinant *= permutationSign(symbolic.get(ADDRESS, KLU_SYMBOLIC_Q), factor.n)
        val scales = numeric.get(ADDRESS, KLU_NUMERIC_RS)
        if (scales.address() != 0L) {
            val boundedScales = scales.reinterpret(factor.n.toLong() * JAVA_DOUBLE.byteSize())
            for (i in 0 until factor.n) determinant /= boundedScales.getAtIndex(JAVA_DOUBLE, i.toLong())
        }
        return determinant
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

    private fun permutationSign(permutation: MemorySegment, n: Int): Double {
        val boundedPermutation = permutation.reinterpret(n.toLong() * JAVA_INT.byteSize())
        var inversions = 0
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                if (boundedPermutation.getAtIndex(JAVA_INT, i.toLong()) >
                    boundedPermutation.getAtIndex(JAVA_INT, j.toLong())
                ) {
                    inversions++
                }
            }
        }
        return if (inversions and 1 == 0) 1.0 else -1.0
    }
}

internal class KluFactor(
    val n: Int,
    val arena: Arena,
    val common: MemorySegment,
    val symbolicHolder: MemorySegment,
    val numericHolder: MemorySegment,
) {
    val symbolic: MemorySegment get() = symbolicHolder.get(ADDRESS, 0).reinterpret(KLU_SYMBOLIC_BYTES)
    val numeric: MemorySegment get() = numericHolder.get(ADDRESS, 0).reinterpret(KLU_NUMERIC_BYTES)

    val nnz: Int
        get() {
            return numeric.get(JAVA_INT, KLU_NUMERIC_LNZ) + numeric.get(JAVA_INT, KLU_NUMERIC_UNZ)
        }
}

internal val KLU_SONAMES = listOf("libklu.so.2", "libklu.2.dylib")

private const val KLU_COMMON_BYTES = 160L
private const val KLU_SYMBOLIC_BYTES = 96L
private const val KLU_NUMERIC_BYTES = 168L
private const val KLU_COMMON_SCALE = 48L
private const val KLU_COMMON_STATUS = 76L
private const val KLU_SYMBOLIC_Q = 56L
private const val KLU_NUMERIC_LNZ = 8L
private const val KLU_NUMERIC_UNZ = 12L
private const val KLU_NUMERIC_PNUM = 24L
private const val KLU_NUMERIC_UDIAG = 88L
private const val KLU_NUMERIC_RS = 96L
private const val KLU_SCALE_NONE = 0
private const val KLU_SCALE_MAX = 2
private const val KLU_SINGULAR = 1
