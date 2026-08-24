package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.internal.host.FfmLibrary
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
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
    )

    /** The configured scaling Control array, built lazily once the library resolves. */
    private val scaledControl: MemorySegment? by lazy { buildControl(config.scaling) }

    /** The unscaled Control array used for factors requested without equilibration. */
    private val unscaledControl: MemorySegment? by lazy { buildControl(UmfpackScaling.NONE) }

    /** The refinement steps a solve will run, or null when no Control array could be built. */
    val refinementSteps: Double? get() = scaledControl?.getAtIndex(JAVA_DOUBLE, IRSTEP.toLong())

    /** The pivot tolerance, for the test that the array holds UMFPACK's defaults and not zeros. */
    val pivotTolerance: Double? get() = scaledControl?.getAtIndex(JAVA_DOUBLE, PIVOT_TOLERANCE.toLong())

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
        return Handles(symbolic, numeric, solve, freeSymbolic, freeNumeric, defaults)
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
