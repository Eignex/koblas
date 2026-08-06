package com.eignex.koblas.umfpack

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/**
 * The UMFPACK subset this backend dispatches to, bound with `java.lang.foreign` downcalls.
 *
 * UMFPACK is SuiteSparse's sparse LU, and the reason it is the host library worth binding first: it is the
 * de facto standard for unsymmetric sparse direct solves, and its stated input format *is* koblas's. The
 * `umfpack.h` preconditions — `Ap[0] == 0`, `Ap[j] <= Ap[j+1]`, row indices ascending within a column with
 * no duplicates, 0-based and in range — are the invariant `SparseMatrix` already validates in its
 * constructor, so a `SparseMatrix` crosses to `umfpack_di_*` with no repacking at all. Checked against the
 * 7.x headers rather than assumed.
 *
 * The `di` family is the `int32_t` index / `double` value one, matching koblas's `IntArray` and
 * `DoubleArray`. The `dl` family takes `int64_t` and would need a widening copy; that is the variant to add
 * if a caller ever has more than 2^31 nonzeros.
 *
 * ### Resolution happens in two stages, and the split is load-bearing
 *
 * [libraryPresent] does nothing but `dlopen`. [available] additionally creates the downcall handles, and only
 * when something first actually calls UMFPACK.
 *
 * They are separate because binding is expensive in a way that bit. `Linker.downcallHandle` generates
 * method-handle forms and is stack-hungry, and doing six of them from inside backend discovery — which runs
 * at whatever depth the first `koblas` read happens to sit at — threw `StackOverflowError` inside a full test
 * suite while succeeding when the same code ran alone. The first version caught `Throwable`, so a blown stack
 * was reported as "SuiteSparse is not installed" and the backend silently never registered, on a machine that
 * had it. Deferring the binds to a call site, where the stack is shallow, removes the cause; narrowing the
 * catch removes the disguise.
 *
 * Hence nothing here catches `Error` broadly. A `StackOverflowError` is not a missing library, and treating
 * the two alike is exactly how that defect hid. Only `IllegalArgumentException` and `UnsatisfiedLinkError`
 * from the `dlopen` count as absence.
 *
 * Nothing is bundled and nothing is linked: `libumfpack` is looked up by soname, so this backend exists
 * exactly on machines that have SuiteSparse and is absent otherwise, with koblas's portable `SparseLu`
 * taking over.
 *
 * Calls go through `invokeWithArguments` rather than `invokeExact` for the reason `HostBlasCalls` documents:
 * `invokeExact` is signature-polymorphic, Kotlin boxes its arguments instead of emitting it, and the native
 * side then reads garbage.
 */
internal object UmfpackCalls {

    /** `umfpack_di_solve`'s `sys` selector: `Ax = b`. */
    const val SYS_A = 0

    /** `umfpack_di_solve`'s `sys` selector: `Aᵀx = b`. For real matrices this is the plain transpose. */
    const val SYS_AT = 1

    /** `Info` array length (`UMFPACK_INFO`). */
    const val INFO = 90

    /** `umfpack_di_*` success. */
    const val OK = 0

    /** `UMFPACK_WARNING_singular_matrix`: a factorization was produced, but the matrix is singular. */
    const val WARNING_SINGULAR = 1

    private val linker = Linker.nativeLinker()

    // UMFPACK reads and writes the caller's arrays; critical hands a Kotlin array over pinned rather than
    // copied, which is what makes a per-solve call worth making at all.
    private val critical = Linker.Option.critical(true)

    private val lookup: SymbolLookup? by lazy { openUmfpack() }

    @Volatile
    private var bindingFailure: String? = null

    /**
     * The bound handles, created on first use.
     *
     * `by lazy` rather than an `init` block so the binding happens in whatever frame first *calls* UMFPACK,
     * not whatever frame first *mentions* this object. That distinction is the whole fix.
     */
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val symbolic: MethodHandle,
        val numeric: MethodHandle,
        val solve: MethodHandle,
        val freeSymbolic: MethodHandle?,
        val freeNumeric: MethodHandle,
        val determinant: MethodHandle?,
    )

    /**
     * Whether `libumfpack` can be opened at all — a `dlopen`, and nothing else.
     *
     * Cheap and shallow enough to call from backend discovery, which is the point: it answers "is SuiteSparse
     * installed" without creating a single downcall handle.
     */
    val libraryPresent: Boolean get() = lookup != null

    /**
     * Whether UMFPACK is usable: the library opened *and* its symbols bound.
     *
     * Reading this binds the handles, so prefer [libraryPresent] wherever the answer is only needed to decide
     * whether to offer the backend at all.
     */
    val available: Boolean get() = handles != null

    /** Why UMFPACK is unusable, or null when it is usable. For diagnostics, not control flow. */
    val unavailableReason: String?
        get() = when {
            lookup == null -> "libumfpack could not be opened; SuiteSparse does not appear to be installed"
            handles == null -> "libumfpack opened but its symbols did not bind: ${bindingFailure ?: "unknown"}"
            else -> null
        }

    /**
     * Opens `libumfpack` by soname, newest first.
     *
     * Versioned sonames lead because a bare `libumfpack.so` is the development symlink and need not exist on
     * a runtime-only machine. The dependent SuiteSparse libraries (amd, colamd, cholmod, suitesparseconfig)
     * arrive through `DT_NEEDED` rather than needing lookups of their own.
     */
    private fun openUmfpack(): SymbolLookup? {
        for (soname in listOf("libumfpack.so.6", "libumfpack.so.5", "libumfpack.so", "libumfpack.dylib")) {
            try {
                return SymbolLookup.libraryLookup(soname, Arena.global())
            } catch (_: IllegalArgumentException) {
                continue // not on this machine
            } catch (_: UnsatisfiedLinkError) {
                continue // present but unloadable
            }
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
        if (symbolic == null || numeric == null || solve == null || freeNumeric == null) {
            bindingFailure = "missing one of umfpack_di_symbolic, _numeric, _solve or _free_numeric"
            return null
        }
        return Handles(symbolic, numeric, solve, freeSymbolic, freeNumeric, determinant)
    }

    /**
     * `int f(...)` taking [ints] leading `int` arguments then [pointers] pointers.
     *
     * Every routine bound here has that shape, so one builder covers them all; spelling out nine `ADDRESS`es
     * per call was noise that obscured which signature was which.
     */
    // The spread copies the array, which detekt rightly flags in a hot path. This is not one: it runs six
    // times in the process's life, once per bound symbol, and the alternative is nine explicit ADDRESS
    // arguments per call site — which is what this replaced, because it hid which signature was which.
    @Suppress("SpreadOperator")
    private fun intsThenPointers(ints: Int, pointers: Int): FunctionDescriptor {
        val args = Array(ints + pointers) { if (it < ints) JAVA_INT else ADDRESS }
        return FunctionDescriptor.of(JAVA_INT, *args)
    }

    private fun bind(found: SymbolLookup, name: String, descriptor: FunctionDescriptor): MethodHandle? {
        val address = found.find(name).orElse(null) ?: return null
        return linker.downcallHandle(address, descriptor, critical)
    }

    private fun handlesOrThrow(): Handles =
        checkNotNull(handles) { "umfpack is not available: ${unavailableReason ?: "unknown reason"}" }

    /** `umfpack_di_symbolic`: the pattern analysis, which does not look at [values]. */
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

    /** `umfpack_di_numeric`: the numeric factorization, given a symbolic analysis. */
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

    /**
     * `umfpack_di_solve`.
     *
     * Takes the matrix again alongside the factors, which is not redundancy: UMFPACK performs its own
     * iterative refinement by default and the residual needs the original `A`. That is also why the
     * factorization keeps its `SparseMatrix` alive.
     */
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
        MemorySegment.NULL,
        info,
    ) as Int

    /** `umfpack_di_get_determinant`: mantissa into [mx], base-10 exponent into [ex]. */
    fun determinant(mx: MemorySegment, ex: MemorySegment, numeric: MemorySegment, info: MemorySegment): Int {
        val handle = handlesOrThrow().determinant ?: return OK
        return handle.invokeWithArguments(mx, ex, numeric, info) as Int
    }

    /** `umfpack_di_free_symbolic`; the analysis is not needed once the numeric factors exist. */
    fun freeSymbolic(symbolicHolder: MemorySegment) {
        handles?.freeSymbolic?.invokeWithArguments(symbolicHolder)
    }

    /** `umfpack_di_free_numeric`. Not calling this leaks whatever UMFPACK malloc'd for the factors. */
    fun freeNumeric(numericHolder: MemorySegment) {
        handles?.freeNumeric?.invokeWithArguments(numericHolder)
    }
}
