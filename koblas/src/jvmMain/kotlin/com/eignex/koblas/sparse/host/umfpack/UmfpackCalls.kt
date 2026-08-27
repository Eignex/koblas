package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.sparse.internal.transposeRaw
import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

/**
 * The LP64 umfpack_di family, whose `int32_t` indices and `double` values match IntArray and DoubleArray;
 * the dl family takes `int64_t` and would need widening copies.
 */
internal class UmfpackCalls(private val config: UmfpackConfig) {

    /** Opened lazily, since a binding constructed during discovery must not load a library to exist. */
    private val library: FfmLibrary by lazy {
        FfmLibrary.open(config.libraryPath?.let(::listOf) ?: UMFPACK_SONAMES, KEY_SYMBOL, "libumfpack")
    }

    @Volatile
    private var bindingFailure: String? = null

    /** Bound lazily because `Linker.downcallHandle` is stack-hungry and discovery runs at arbitrary depth. */
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val symbolic: MethodHandle,
        val numeric: MethodHandle,
        val solve: MethodHandle,
        val freeSymbolic: MethodHandle,
        val freeNumeric: MethodHandle,
        val defaults: MethodHandle,
        val getLunz: MethodHandle?,
        val getNumeric: MethodHandle?,
    )

    /** The configured scaling Control array, built lazily once the library resolves. */
    private val scaledControl: MemorySegment? by lazy { buildControl(config.scaling) }

    /** The unscaled Control array used for factors requested without equilibration. */
    private val unscaledControl: MemorySegment? by lazy { buildControl(UmfpackScaling.NONE) }

    /** The refinement steps a solve will run, or null when no Control array could be built. */
    val refinementSteps: Double? get() = scaledControl?.getAtIndex(JAVA_DOUBLE, IRSTEP.toLong())

    /** The pivot tolerance, for the test that the array holds UMFPACK's defaults and not zeros. */
    val pivotTolerance: Double? get() = scaledControl?.getAtIndex(JAVA_DOUBLE, PIVOT_TOLERANCE.toLong())

    /** The native scaling selector written into the effective Control array. */
    val scaling: Double? get() = scaledControl?.getAtIndex(JAVA_DOUBLE, SCALE.toLong())

    /** The Control array whose scaling agrees with one factorization request. */
    fun control(equilibrate: Boolean): MemorySegment? = if (equilibrate) scaledControl else unscaledControl

    private fun buildControl(scaling: UmfpackScaling): MemorySegment? {
        val defaults = (handles ?: return null).defaults
        // The global arena keeps this read-only array alive past every solve.
        val control = Arena.global().allocate(JAVA_DOUBLE, CONTROL.toLong())
        defaults.invokeExact(control) as Unit
        control.setAtIndex(JAVA_DOUBLE, IRSTEP.toLong(), config.iterativeRefinementSteps.toDouble())
        control.setAtIndex(JAVA_DOUBLE, PIVOT_TOLERANCE.toLong(), config.pivotTolerance)
        control.setAtIndex(JAVA_DOUBLE, SCALE.toLong(), scaling.nativeValue)
        return control
    }

    /** Whether a libumfpack carrying the di family opened, which creates no downcall handle. */
    val libraryPresent: Boolean get() = library.present

    /** Whether the library opened and its symbols bound. Reading this binds the handles. */
    val available: Boolean get() = handles != null

    /** Why UMFPACK is unusable, or null when it is usable. For diagnostics, not control flow. */
    val unavailableReason: String?
        get() = when {
            !library.present -> "libumfpack could not be opened; SuiteSparse does not appear to be installed"
            handles == null -> "libumfpack opened but its symbols did not bind: ${bindingFailure ?: "unknown"}"
            else -> null
        }

    private fun bindAll(): Handles? {
        if (!library.present) return null
        // int umfpack_di_symbolic(int n_row, int n_col, const int Ap[], const int Ai[], const double Ax[],
        //                         void **Symbolic, const double Control[], double Info[])
        val symbolic = bind("umfpack_di_symbolic", intsThenPointers(ints = 2, pointers = 6))
        // int umfpack_di_numeric(const int Ap[], const int Ai[], const double Ax[], void *Symbolic,
        //                        void **Numeric, const double Control[], double Info[])
        val numeric = bind("umfpack_di_numeric", intsThenPointers(ints = 0, pointers = 7))
        // int umfpack_di_solve(int sys, const int Ap[], const int Ai[], const double Ax[], double X[],
        //                      const double B[], void *Numeric, const double Control[], double Info[])
        val solve = bind("umfpack_di_solve", intsThenPointers(ints = 1, pointers = 8))
        val freeSymbolic = bind("umfpack_di_free_symbolic", FunctionDescriptor.ofVoid(ADDRESS))
        val freeNumeric = bind("umfpack_di_free_numeric", FunctionDescriptor.ofVoid(ADDRESS))
        // void umfpack_di_defaults(double Control[UMFPACK_CONTROL])
        val defaults = bind("umfpack_di_defaults", FunctionDescriptor.ofVoid(ADDRESS))
        // Every one is required. An optional binding degrades silently instead: without _defaults the solve
        // keeps UMFPACK's iterative refinement on, and without _free_symbolic the analysis leaks.
        if (symbolic == null || numeric == null || solve == null || freeNumeric == null ||
            freeSymbolic == null || defaults == null
        ) {
            bindingFailure = "the host libumfpack lacks one of the umfpack_di_ symbols koblas binds"
            return null
        }
        // int umfpack_di_get_lunz(int *lnz, int *unz, int *n_row, int *n_col, int *nz_udiag, void *Numeric)
        val getLunz = bind("umfpack_di_get_lunz", intsThenPointers(ints = 0, pointers = 6))
        // int umfpack_di_get_numeric(Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx, do_recip, Rs, void *Numeric)
        val getNumeric = bind("umfpack_di_get_numeric", intsThenPointers(ints = 0, pointers = 12))
        // These two are optional, unlike the rest: without them a factorization still solves, and only
        // reading its factors is unavailable.
        return Handles(symbolic, numeric, solve, freeSymbolic, freeNumeric, defaults, getLunz, getNumeric)
    }

    /** `int f(...)` taking [ints] leading `int` arguments then [pointers] pointers. */
    // The spread copies the array, once per bound symbol and off any hot path.
    @Suppress("SpreadOperator")
    private fun intsThenPointers(ints: Int, pointers: Int): FunctionDescriptor {
        val args = Array(ints + pointers) { if (it < ints) JAVA_INT else ADDRESS }
        return FunctionDescriptor.of(JAVA_INT, *args)
    }

    private fun bind(name: String, descriptor: FunctionDescriptor): MethodHandle? =
        library.handleOrNull(name, descriptor)

    /**
     * `L`, `U`, the two permutations and the row scaling from [numeric], or null when this libumfpack lacks
     * the extraction symbols.
     *
     * `L` arrives in row form and is transposed to reach CSC. The scaling arrives as a multiplier or a
     * divisor depending on `do_recip`, and is normalised to a multiplier here so the identity
     * `L·U = P·diag(rowScaling)·A·Q` holds without asking which it was.
     */
    fun extractFactors(numeric: MemorySegment, order: Int): UmfpackFactors? {
        val bound = handles ?: return null
        val lunz = bound.getLunz ?: return null
        val getNumeric = bound.getNumeric ?: return null
        return Arena.ofConfined().use { arena ->
            val sizes = arena.allocate(JAVA_INT, 5)
            val slot = { index: Long -> sizes.asSlice(index * Int.SIZE_BYTES, Int.SIZE_BYTES.toLong()) }
            if (lunz.invokeExact(slot(0), slot(1), slot(2), slot(3), slot(4), numeric) as Int != OK) {
                return@use null
            }
            val lNonzeros = sizes.getAtIndex(JAVA_INT, 0)
            val uNonzeros = sizes.getAtIndex(JAVA_INT, 1)
            val lRowPtr = arena.allocate(JAVA_INT, order + 1L)
            val lColIdx = arena.allocate(JAVA_INT, maxOf(lNonzeros, 1).toLong())
            val lValues = arena.allocate(JAVA_DOUBLE, maxOf(lNonzeros, 1).toLong())
            val uColPtr = arena.allocate(JAVA_INT, order + 1L)
            val uRowIdx = arena.allocate(JAVA_INT, maxOf(uNonzeros, 1).toLong())
            val uValues = arena.allocate(JAVA_DOUBLE, maxOf(uNonzeros, 1).toLong())
            val rowPerm = arena.allocate(JAVA_INT, order.toLong())
            val colPerm = arena.allocate(JAVA_INT, order.toLong())
            val reciprocal = arena.allocate(JAVA_INT, 1)
            val scaling = arena.allocate(JAVA_DOUBLE, order.toLong())
            val status = getNumeric.invokeExact(
                lRowPtr, lColIdx, lValues, uColPtr, uRowIdx, uValues,
                rowPerm, colPerm, MemorySegment.NULL, reciprocal, scaling, numeric,
            ) as Int
            if (status != OK) return@use null
            UmfpackFactors(
                lower = read(lRowPtr, lColIdx, lValues, order, lNonzeros, transposed = true),
                upper = read(uColPtr, uRowIdx, uValues, order, uNonzeros, transposed = false),
                rowOrder = ints(rowPerm, order),
                columnOrder = ints(colPerm, order),
                rowScaling = doubles(scaling, order).let { scale ->
                    // do_recip false means UMFPACK divided by these, so the multiplier is the reciprocal.
                    if (reciprocal.getAtIndex(JAVA_INT, 0) != 0) scale else DoubleArray(order) { 1.0 / scale[it] }
                },
            )
        }
    }

    private fun read(
        pointers: MemorySegment,
        indices: MemorySegment,
        values: MemorySegment,
        order: Int,
        nonzeros: Int,
        transposed: Boolean,
    ): F64SparseMatrix {
        val ptr = ints(pointers, order + 1)
        val idx = ints(indices, nonzeros)
        val entries = doubles(values, nonzeros)
        // Row form is the transpose in CSC, and transposing sorts the rows a library need not have sorted.
        return if (transposed) {
            transposeRaw(order, order, ptr, idx, entries)
        } else {
            transposeRaw(
                order,
                order,
                ptr,
                idx,
                entries,
            ).let { transposeRaw(order, order, it.colPtr, it.rowIdx, it.values) }
        }
    }

    private fun ints(segment: MemorySegment, count: Int): IntArray {
        val out = IntArray(count)
        MemorySegment.copy(segment, JAVA_INT, 0L, out, 0, count)
        return out
    }

    private fun doubles(segment: MemorySegment, count: Int): DoubleArray {
        val out = DoubleArray(count)
        MemorySegment.copy(segment, JAVA_DOUBLE, 0L, out, 0, count)
        return out
    }

    private fun handlesOrThrow(): Handles =
        checkNotNull(handles) { "umfpack is not available: ${unavailableReason ?: "unknown reason"}" }

    /** The pattern analysis, which does not look at [values]. */
    fun symbolic(
        n: Int,
        colPtr: MemorySegment,
        rowIdx: MemorySegment,
        values: MemorySegment,
        symbolicOut: MemorySegment,
        control: MemorySegment?,
        info: MemorySegment,
    ): Int = handlesOrThrow().symbolic.invokeExact(
        n,
        n,
        colPtr,
        rowIdx,
        values,
        symbolicOut,
        control ?: MemorySegment.NULL,
        info,
    ) as Int

    /** The numeric factorization, given a symbolic analysis. */
    fun numeric(
        colPtr: MemorySegment,
        rowIdx: MemorySegment,
        values: MemorySegment,
        symbolic: MemorySegment,
        numericOut: MemorySegment,
        control: MemorySegment?,
        info: MemorySegment,
    ): Int = handlesOrThrow().numeric.invokeExact(
        colPtr,
        rowIdx,
        values,
        symbolic,
        numericOut,
        control ?: MemorySegment.NULL,
        info,
    ) as Int

    /** Takes Ap, Ai and Ax alongside the factors, so a caller must keep the matrix alive to solve. */
    @Suppress("LongParameterList") // the umfpack_di_solve signature
    fun solve(
        sys: Int,
        colPtr: MemorySegment,
        rowIdx: MemorySegment,
        values: MemorySegment,
        x: MemorySegment,
        b: MemorySegment,
        numeric: MemorySegment,
        control: MemorySegment?,
        info: MemorySegment,
    ): Int = handlesOrThrow().solve.invokeExact(
        sys,
        colPtr,
        rowIdx,
        values,
        x,
        b,
        numeric,
        control ?: MemorySegment.NULL,
        info,
    ) as Int

    /** The analysis is not needed once the numeric factors exist. */
    fun freeSymbolic(symbolicHolder: MemorySegment) {
        // Resolved before the call because a safe-call chain makes the result nullable, and `as Unit?` is
        // the boxed Unit rather than void, which is not the descriptor.
        val free = handles?.freeSymbolic ?: return
        free.invokeExact(symbolicHolder) as Unit
    }

    /** Not calling this leaks whatever UMFPACK malloc'd for the factors. */
    fun freeNumeric(numericHolder: MemorySegment) {
        val free = handles?.freeNumeric ?: return
        free.invokeExact(numericHolder) as Unit
    }
}

private val UmfpackScaling.nativeValue: Double
    get() = when (this) {
        UmfpackScaling.NONE -> 0.0
        UmfpackScaling.SUM -> 1.0
        UmfpackScaling.MAX -> 2.0
    }
