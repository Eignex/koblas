package com.eignex.koblas.sparse.host.superlu

import com.eignex.koblas.internal.backend.nativeLibraryPaths
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemoryLayout
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/** The public LP64 ABI exported by SuperLU 7. */
internal object SuperluCalls {
    private val linker: Linker? = runCatching { Linker.nativeLinker() }.getOrNull()
    private val critical = Linker.Option.critical(true)
    private val lookup: SymbolLookup? by lazy { openSuperlu() }

    @Volatile private var bindingFailure: String? = null
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val createCompCol: MethodHandle,
        val createDense: MethodHandle,
        val defaults: MethodHandle,
        val permutation: MethodHandle,
        val preorder: MethodHandle,
        val factor: MethodHandle,
        val expert: MethodHandle,
        val solve: MethodHandle,
        val refine: MethodHandle,
        val diagonal: MethodHandle,
        val destroyStore: MethodHandle,
        val destroyPermuted: MethodHandle,
        val destroySuperNode: MethodHandle,
        val destroyCompCol: MethodHandle,
        val statInit: MethodHandle,
        val statFree: MethodHandle,
    )

    val libraryPresent: Boolean get() = linker != null && lookup != null
    val available: Boolean get() = handles != null
    val unavailableReason: String?
        get() =
            when {
                lookup == null -> "SuperLU 7 LP64 could not be opened"

                handles == null ->
                    "SuperLU opened but its v7 symbols did not bind: ${bindingFailure ?: "unknown"}"

                else -> null
            }

    private fun openSuperlu(): SymbolLookup? {
        val paths = nativeLibraryPaths("koblas.superlu.path", "KOBLAS_SUPERLU_PATH", SUPERLU_SONAMES)
        for (soname in paths) {
            val opened =
                try {
                    SymbolLookup.libraryLookup(
                        soname,
                        Arena.global(),
                    )
                } catch (_: IllegalArgumentException) {
                    continue
                } catch (_: UnsatisfiedLinkError) {
                    continue
                }
            if (opened.find("dgstrf").isPresent) return opened
        }
        return null
    }

    private class MissingSymbol : RuntimeException()

    private fun bindAll(): Handles? = try {
        val found = lookup ?: return null

        fun bind(name: String, result: MemoryLayout, vararg args: MemoryLayout): MethodHandle {
            val address = found.find(name).orElse(null) ?: throw MissingSymbol()
            return checkNotNull(linker).downcallHandle(
                address,
                FunctionDescriptor.of(result, *args),
                critical,
            )
        }

        fun bindVoid(name: String, vararg args: MemoryLayout): MethodHandle {
            val address = found.find(name).orElse(null) ?: throw MissingSymbol()
            return checkNotNull(linker).downcallHandle(
                address,
                FunctionDescriptor.ofVoid(*args),
                critical,
            )
        }
        return Handles(
            bindVoid(
                "dCreate_CompCol_Matrix",
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
            ),
            bindVoid(
                "dCreate_Dense_Matrix",
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
                JAVA_INT,
            ),
            bindVoid("set_default_options", ADDRESS),
            bindVoid("get_perm_c", JAVA_INT, ADDRESS, ADDRESS),
            bindVoid("sp_preorder", ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
            bindVoid(
                "dgstrf",
                ADDRESS,
                ADDRESS,
                JAVA_INT,
                JAVA_INT,
                ADDRESS,
                ADDRESS,
                JAVA_INT,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
            ),
            bindVoid(
                "dgssvx",
                ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
                ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
                ADDRESS, ADDRESS,
            ),
            bindVoid(
                "dgstrs",
                JAVA_INT,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
            ),
            bindVoid(
                "dgsrfs",
                JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
                ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
            ),
            bindVoid(
                "dGetDiagU",
                ADDRESS,
                ADDRESS,
            ),
            bindVoid("Destroy_SuperMatrix_Store", ADDRESS),
            bindVoid("Destroy_CompCol_Permuted", ADDRESS),
            bindVoid("Destroy_SuperNode_Matrix", ADDRESS),
            bindVoid("Destroy_CompCol_Matrix", ADDRESS),
            bindVoid("StatInit", ADDRESS),
            bindVoid("StatFree", ADDRESS),
        )
    } catch (_: MissingSymbol) {
        bindingFailure = "the library lacks one of the SuperLU 7 symbols koblas binds"
        null
    }

    private fun handlesOrThrow(): Handles = checkNotNull(handles) { "SuperLU is not available: $unavailableReason" }

    private fun MethodHandle.call(vararg arguments: Any?) {
        invokeWithArguments(*arguments)
    }

    fun factorize(
        n: Int,
        values: DoubleArray,
        rowIdx: IntArray,
        colPtr: IntArray,
        @Suppress("UNUSED_PARAMETER") equilibrate: Boolean,
        arena: Arena,
    ): SuperluFactor? {
        val h = handlesOrThrow()
        Arena.ofConfined().use { scratch ->
            val a = scratch.allocate(SUPER_MATRIX)
            val ac = scratch.allocate(SUPER_MATRIX)
            val options = scratch.allocate(OPTIONS)
            val etree = scratch.allocate(JAVA_INT, n.toLong())
            val factor =
                SuperluFactor(
                    arena.allocate(SUPER_MATRIX),
                    arena.allocate(SUPER_MATRIX),
                    arena.allocate(JAVA_INT, n.toLong()),
                    arena.allocate(JAVA_INT, n.toLong()),
                )
            val glu =
                scratch.allocate(
                    GLOBAL_LU,
                )
            val stat = scratch.allocate(STAT)
            val info = scratch.allocate(JAVA_INT)
            h.createCompCol.call(
                a,
                n,
                n,
                rowIdx.size,
                MemorySegment.ofArray(values),
                MemorySegment.ofArray(rowIdx),
                MemorySegment.ofArray(colPtr),
                SLU_NC,
                SLU_D,
                SLU_GE,
            )
            h.defaults.call(
                options,
            )
            options.set(JAVA_INT, EQUIL_OFFSET, NO)
            options.set(JAVA_INT, PRINT_STAT_OFFSET, NO)
            h.permutation.call(options.get(JAVA_INT, COL_PERM_OFFSET), a, factor.permC)
            h.preorder.call(options, a, factor.permC, etree, ac)
            h.statInit.call(stat)
            try {
                h.factor.call(
                    options,
                    ac,
                    PANEL_SIZE,
                    RELAX,
                    etree,
                    MemorySegment.NULL,
                    0,
                    factor.permC,
                    factor.permR,
                    factor.l,
                    factor.u,
                    glu,
                    stat,
                    info,
                )
            } finally {
                h.statFree.call(stat)
                h.destroyPermuted.call(ac)
                h.destroyStore.call(a)
            }
            if (info.get(JAVA_INT, 0) != 0) {
                free(factor)
                return null
            }
            return factor
        }
    }

    fun factorizeExpert(
        n: Int,
        values: DoubleArray,
        rowIdx: IntArray,
        colPtr: IntArray,
        equilibrate: Boolean,
        arena: Arena,
    ): SuperluFactor? {
        val h = handlesOrThrow()
        Arena.ofConfined().use { scratch ->
            val a = scratch.allocate(SUPER_MATRIX)
            val options = scratch.allocate(OPTIONS)
            val etree = scratch.allocate(JAVA_INT, n.toLong())
            val equed = scratch.allocate(JAVA_BYTE)
            val rowScale = DoubleArray(n)
            val columnScale = DoubleArray(n)
            val factor =
                SuperluFactor(
                    arena.allocate(SUPER_MATRIX),
                    arena.allocate(SUPER_MATRIX),
                    arena.allocate(JAVA_INT, n.toLong()),
                    arena.allocate(JAVA_INT, n.toLong()),
                    rowScale = rowScale,
                    columnScale = columnScale,
                    matrixValues = values,
                    rowIdx = rowIdx,
                    colPtr = colPtr,
                )
            val glu = scratch.allocate(GLOBAL_LU)
            val stat = scratch.allocate(STAT)
            val memory = scratch.allocate(MEMORY)
            val info = scratch.allocate(JAVA_INT)
            val b = scratch.allocate(SUPER_MATRIX)
            val x = scratch.allocate(SUPER_MATRIX)
            val dummy = scratch.allocate(JAVA_DOUBLE)
            val reciprocalPivotGrowth = scratch.allocate(JAVA_DOUBLE)
            val condition = scratch.allocate(JAVA_DOUBLE)
            val forwardError = scratch.allocate(JAVA_DOUBLE)
            val backwardError = scratch.allocate(JAVA_DOUBLE)
            h.createCompCol.call(
                a, n, n, rowIdx.size, MemorySegment.ofArray(values), MemorySegment.ofArray(rowIdx),
                MemorySegment.ofArray(colPtr), SLU_NC, SLU_D, SLU_GE,
            )
            h.defaults.call(options)
            options.set(JAVA_INT, EQUIL_OFFSET, if (equilibrate) YES else NO)
            options.set(JAVA_INT, ITER_REFINE_OFFSET, SLU_DOUBLE)
            options.set(JAVA_INT, PIVOT_GROWTH_OFFSET, YES)
            options.set(JAVA_INT, CONDITION_NUMBER_OFFSET, YES)
            options.set(JAVA_INT, PRINT_STAT_OFFSET, NO)
            h.createDense.call(b, n, 0, dummy, n, SLU_DN, SLU_D, SLU_GE)
            h.createDense.call(x, n, 0, dummy, n, SLU_DN, SLU_D, SLU_GE)
            h.statInit.call(stat)
            try {
                h.expert.call(
                    options, a, factor.permC, factor.permR, etree, equed, MemorySegment.ofArray(rowScale),
                    MemorySegment.ofArray(columnScale), factor.l, factor.u, MemorySegment.NULL, 0, b, x,
                    reciprocalPivotGrowth, condition, forwardError, backwardError, glu, memory, stat, info,
                )
            } finally {
                h.statFree.call(stat)
                h.destroyStore.call(a)
                h.destroyStore.call(b)
                h.destroyStore.call(x)
            }
            if (info.get(JAVA_INT, 0) != 0) {
                free(factor)
                return null
            }
            factor.equed = equed.get(JAVA_BYTE, 0)
            return factor
        }
    }

    fun solve(factor: SuperluFactor, rhs: MemorySegment, transpose: Boolean) {
        val h = handlesOrThrow()
        Arena.ofConfined().use { scratch ->
            val b =
                scratch.allocate(
                    SUPER_MATRIX,
                )
            val originalB = scratch.allocate(SUPER_MATRIX)
            val originalRhs = scratch.allocate(JAVA_DOUBLE, factor.n.toLong())
            val stat = scratch.allocate(STAT)
            val info = scratch.allocate(JAVA_INT)
            val ferr = scratch.allocate(JAVA_DOUBLE)
            val berr = scratch.allocate(JAVA_DOUBLE)
            originalRhs.copyFrom(rhs)
            h.createDense.call(b, factor.n, 1, rhs, factor.n, SLU_DN, SLU_D, SLU_GE)
            h.createDense.call(originalB, factor.n, 1, originalRhs, factor.n, SLU_DN, SLU_D, SLU_GE)
            h.statInit.call(stat)
            try {
                h.solve.call(
                    if (transpose) TRANS else NOTRANS,
                    factor.l,
                    factor.u,
                    factor.permC,
                    factor.permR,
                    b,
                    stat,
                    info,
                )
                factor.matrixValues?.let { values ->
                    val a = scratch.allocate(SUPER_MATRIX)
                    h.createCompCol.call(
                        a, factor.n, factor.n, checkNotNull(factor.rowIdx).size, MemorySegment.ofArray(values),
                        MemorySegment.ofArray(
                            checkNotNull(factor.rowIdx),
                        ),
                        MemorySegment.ofArray(checkNotNull(factor.colPtr)),
                        SLU_NC, SLU_D, SLU_GE,
                    )
                    try {
                        h.refine.call(
                            if (transpose) TRANS else NOTRANS, a, factor.l, factor.u, factor.permC, factor.permR,
                            MemorySegment.ofArray(byteArrayOf(factor.equed)), MemorySegment.ofArray(factor.rowScale),
                            MemorySegment.ofArray(factor.columnScale), originalB, b, ferr, berr, stat, info,
                        )
                    } finally {
                        h.destroyStore.call(a)
                    }
                }
            } finally {
                h.statFree.call(stat)
                h.destroyStore.call(b)
                h.destroyStore.call(originalB)
            }
            check(info.get(JAVA_INT, 0) == 0) { "SuperLU solve failed" }
        }
    }

    fun determinant(factor: SuperluFactor): Double {
        val diagonal = DoubleArray(factor.n)
        handlesOrThrow().diagonal.call(factor.l, MemorySegment.ofArray(diagonal))
        var result =
            permutationSign(factor.permC, factor.n) * permutationSign(factor.permR, factor.n)
        for (entry in diagonal) result *= entry
        if (factor.equed == ROW || factor.equed == BOTH) for (scale in factor.rowScale) result /= scale
        if (factor.equed == COL || factor.equed == BOTH) for (scale in factor.columnScale) result /= scale
        return result
    }

    fun scaleRhs(factor: SuperluFactor, rhs: DoubleArray, transpose: Boolean) {
        val scale = if (transpose) factor.columnScale else factor.rowScale
        val applied = if (transpose) {
            factor.equed == COL || factor.equed == BOTH
        } else {
            factor.equed == ROW ||
                factor.equed == BOTH
        }
        if (applied) for (i in rhs.indices) rhs[i] *= scale[i]
    }

    fun unscaleSolution(factor: SuperluFactor, solution: DoubleArray, transpose: Boolean) {
        val scale = if (transpose) factor.rowScale else factor.columnScale
        val applied = if (transpose) {
            factor.equed == ROW || factor.equed == BOTH
        } else {
            factor.equed == COL ||
                factor.equed == BOTH
        }
        if (applied) for (i in solution.indices) solution[i] *= scale[i]
    }

    fun fill(factor: SuperluFactor): Int = storeNnz(factor.l) + storeNnz(factor.u)

    fun free(factor: SuperluFactor) {
        val h = handlesOrThrow()
        if (factor.l.get(ADDRESS, STORE_OFFSET).address() != 0L) h.destroySuperNode.call(factor.l)
        if (factor.u.get(ADDRESS, STORE_OFFSET).address() != 0L) h.destroyCompCol.call(factor.u)
    }

    private fun storeNnz(matrix: MemorySegment): Int = matrix
        .get(
            ADDRESS,
            STORE_OFFSET,
        ).reinterpret(JAVA_INT.byteSize())
        .get(JAVA_INT, 0)

    private fun permutationSign(permutation: MemorySegment, n: Int): Double {
        val seen = BooleanArray(n)
        var sign = 1.0
        for (start in 0 until n) {
            if (!seen[start]) {
                var length = 0
                var at = start
                while (!seen[at]) {
                    seen[at] = true
                    length++
                    at = permutation.getAtIndex(JAVA_INT, at.toLong())
                }
                if (length % 2 == 0) sign = -sign
            }
        }
        return sign
    }
}

internal class SuperluFactor(
    val l: MemorySegment,
    val u: MemorySegment,
    val permC: MemorySegment,
    val permR: MemorySegment,
    var equed: Byte = NOEQUIL,
    val rowScale: DoubleArray = DoubleArray(0),
    val columnScale: DoubleArray = DoubleArray(0),
    val matrixValues: DoubleArray? = null,
    val rowIdx: IntArray? = null,
    val colPtr: IntArray? = null,
) {
    val n: Int get() = l.get(JAVA_INT, NROW_OFFSET)
}

private const val SLU_NC = 0
private const val SLU_DN = 6
private const val SLU_D = 1
private const val SLU_GE = 0
private const val NOTRANS = 0
private const val TRANS = 1
private const val NO = 0
private const val YES = 1
private const val SLU_DOUBLE = 2
private const val NOEQUIL: Byte = 'N'.code.toByte()
private const val ROW: Byte = 'R'.code.toByte()
private const val COL: Byte = 'C'.code.toByte()
private const val BOTH: Byte = 'B'.code.toByte()
private const val PANEL_SIZE = 20
private const val RELAX = 10
private const val EQUIL_OFFSET = 4L
private const val ITER_REFINE_OFFSET = 16L
private const val PIVOT_GROWTH_OFFSET = 36L
private const val CONDITION_NUMBER_OFFSET = 40L
private const val COL_PERM_OFFSET = 8L
private const val PRINT_STAT_OFFSET = 120L
private const val NROW_OFFSET = 12L
private const val STORE_OFFSET = 24L
private val SUPER_MATRIX =
    MemoryLayout.structLayout(
        JAVA_INT,
        JAVA_INT,
        JAVA_INT,
        JAVA_INT,
        JAVA_INT,
        MemoryLayout.paddingLayout(4),
        ADDRESS,
    )
private val OPTIONS =
    MemoryLayout.paddingLayout(
        144,
    )
private val GLOBAL_LU = MemoryLayout.paddingLayout(192)
private val STAT = MemoryLayout.paddingLayout(48)
private val MEMORY = MemoryLayout.paddingLayout(8)
internal val SUPERLU_SONAMES = listOf("libsuperlu.so.7", "libsuperlu.7.dylib")
