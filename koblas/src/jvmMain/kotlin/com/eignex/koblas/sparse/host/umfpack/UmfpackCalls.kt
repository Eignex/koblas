package com.eignex.koblas.sparse.host.umfpack

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

/**
 * The LP64 umfpack_di family, whose `int32_t` indices and `double` values match IntArray and DoubleArray;
 * the dl family takes `int64_t` and would need widening copies.
 */
internal object UmfpackCalls {

    /** Null where this platform has no native linker; see the same field in `HostBlasCalls`. */
    private val linker: Linker? = try {
        Linker.nativeLinker()
    } catch (_: UnsupportedOperationException) {
        null
    }

    // Pins the Kotlin arrays for the call instead of copying them, blocking relocation while it runs.
    private val critical = Linker.Option.critical(true)

    private val lookup: SymbolLookup? by lazy { openUmfpack() }

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
        val determinant: MethodHandle,
        val defaults: MethodHandle,
    )

    /**
     * The Control array every solve passes, holding UMFPACK's defaults with iterative refinement off, so a
     * solve is the triangular solve against the factors. Null until the library resolves.
     */
    private val solveControl: MemorySegment? by lazy { buildSolveControl() }

    /** The refinement steps a solve will run, or null when no Control array could be built. */
    val refinementSteps: Double? get() = solveControl?.getAtIndex(JAVA_DOUBLE, IRSTEP.toLong())

    /** The pivot tolerance, for the test that the array holds UMFPACK's defaults and not zeros. */
    val pivotTolerance: Double? get() = solveControl?.getAtIndex(JAVA_DOUBLE, PIVOT_TOLERANCE.toLong())

    private fun buildSolveControl(): MemorySegment? {
        val defaults = (handles ?: return null).defaults
        // The global arena keeps this read-only array alive past every solve.
        val control = Arena.global().allocate(JAVA_DOUBLE, CONTROL.toLong())
        defaults.invokeWithArguments(control)
        control.setAtIndex(JAVA_DOUBLE, IRSTEP.toLong(), 0.0)
        return control
    }

    /** Whether a libumfpack carrying the di family opened, which creates no downcall handle. */
    val libraryPresent: Boolean get() = linker != null && lookup != null

    /** Whether the library opened and its symbols bound. Reading this binds the handles. */
    val available: Boolean get() = handles != null

    /** Why UMFPACK is unusable, or null when it is usable. For diagnostics, not control flow. */
    val unavailableReason: String?
        get() = when {
            lookup == null -> "libumfpack could not be opened; SuiteSparse does not appear to be installed"
            handles == null -> "libumfpack opened but its symbols did not bind: ${bindingFailure ?: "unknown"}"
            else -> null
        }

    /**
     * Opens the first libumfpack that loads and carries [KEY_SYMBOL]. Only IllegalArgumentException and
     * UnsatisfiedLinkError count as absence, so a StackOverflowError is never read as a missing library.
     */
    private fun openUmfpack(): SymbolLookup? {
        for (soname in nativeLibraryPaths("koblas.umfpack.path", "KOBLAS_UMFPACK_PATH", UMFPACK_SONAMES)) {
            val opened = try {
                SymbolLookup.libraryLookup(soname, Arena.global())
            } catch (_: IllegalArgumentException) {
                continue // not on this machine
            } catch (_: UnsatisfiedLinkError) {
                continue // present but unloadable
            }
            if (opened.find(KEY_SYMBOL).isPresent) return opened
        }
        return null
    }

    private fun bindAll(): Handles? {
        val found = lookup ?: return null
        // int umfpack_di_symbolic(int n_row, int n_col, const int Ap[], const int Ai[], const double Ax[],
        //                         void **Symbolic, const double Control[], double Info[])
        val symbolic = bind(found, "umfpack_di_symbolic", intsThenPointers(ints = 2, pointers = 6))
        // int umfpack_di_numeric(const int Ap[], const int Ai[], const double Ax[], void *Symbolic,
        //                        void **Numeric, const double Control[], double Info[])
        val numeric = bind(found, "umfpack_di_numeric", intsThenPointers(ints = 0, pointers = 7))
        // int umfpack_di_solve(int sys, const int Ap[], const int Ai[], const double Ax[], double X[],
        //                      const double B[], void *Numeric, const double Control[], double Info[])
        val solve = bind(found, "umfpack_di_solve", intsThenPointers(ints = 1, pointers = 8))
        val freeSymbolic = bind(found, "umfpack_di_free_symbolic", FunctionDescriptor.ofVoid(ADDRESS))
        val freeNumeric = bind(found, "umfpack_di_free_numeric", FunctionDescriptor.ofVoid(ADDRESS))
        // int umfpack_di_get_determinant(double *Mx, double *Ex, void *Numeric, double Info[])
        val determinant = bind(found, "umfpack_di_get_determinant", intsThenPointers(ints = 0, pointers = 4))
        // void umfpack_di_defaults(double Control[UMFPACK_CONTROL])
        val defaults = bind(found, "umfpack_di_defaults", FunctionDescriptor.ofVoid(ADDRESS))
        // Every one is required. An optional binding degrades silently instead: without _get_determinant a
        // determinant reads back as zero, which is indistinguishable from singular, without _defaults the
        // solve keeps UMFPACK's iterative refinement on, and without _free_symbolic the analysis leaks.
        if (symbolic == null || numeric == null || solve == null || freeNumeric == null ||
            freeSymbolic == null || determinant == null || defaults == null
        ) {
            bindingFailure = "the host libumfpack lacks one of the umfpack_di_ symbols koblas binds"
            return null
        }
        return Handles(symbolic, numeric, solve, freeSymbolic, freeNumeric, determinant, defaults)
    }

    /** `int f(...)` taking [ints] leading `int` arguments then [pointers] pointers. */
    // The spread copies the array, once per bound symbol and off any hot path.
    @Suppress("SpreadOperator")
    private fun intsThenPointers(ints: Int, pointers: Int): FunctionDescriptor {
        val args = Array(ints + pointers) { if (it < ints) JAVA_INT else ADDRESS }
        return FunctionDescriptor.of(JAVA_INT, *args)
    }

    /** Calls on the result use `invokeWithArguments`, since Kotlin cannot emit signature-polymorphic `invokeExact`. */
    private fun bind(found: SymbolLookup, name: String, descriptor: FunctionDescriptor): MethodHandle? {
        val downcall = linker ?: return null
        val address = found.find(name).orElse(null) ?: return null
        return downcall.downcallHandle(address, descriptor, critical)
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
        info: MemorySegment,
    ): Int = handlesOrThrow().symbolic.invokeWithArguments(
        n,
        n,
        colPtr,
        rowIdx,
        values,
        symbolicOut,
        MemorySegment.NULL, // Control: NULL takes UMFPACK's defaults
        info,
    ) as Int

    /** The numeric factorization, given a symbolic analysis. */
    fun numeric(
        colPtr: MemorySegment,
        rowIdx: MemorySegment,
        values: MemorySegment,
        symbolic: MemorySegment,
        numericOut: MemorySegment,
        info: MemorySegment,
    ): Int = handlesOrThrow().numeric.invokeWithArguments(
        colPtr,
        rowIdx,
        values,
        symbolic,
        numericOut,
        MemorySegment.NULL,
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
        info: MemorySegment,
    ): Int = handlesOrThrow().solve.invokeWithArguments(
        sys,
        colPtr,
        rowIdx,
        values,
        x,
        b,
        numeric,
        solveControl ?: MemorySegment.NULL,
        info,
    ) as Int

    /** Writes the mantissa into [mx] and the base-10 exponent into [ex]. */
    fun determinant(mx: MemorySegment, ex: MemorySegment, numeric: MemorySegment, info: MemorySegment): Int {
        val handle = handlesOrThrow().determinant
        return handle.invokeWithArguments(mx, ex, numeric, info) as Int
    }

    /** The analysis is not needed once the numeric factors exist. */
    fun freeSymbolic(symbolicHolder: MemorySegment) {
        handles?.freeSymbolic?.invokeWithArguments(symbolicHolder)
    }

    /** Not calling this leaks whatever UMFPACK malloc'd for the factors. */
    fun freeNumeric(numericHolder: MemorySegment) {
        handles?.freeNumeric?.invokeWithArguments(numericHolder)
    }
}
