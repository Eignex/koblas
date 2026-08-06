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
 * constructor, so a `SparseMatrix` crosses to `umfpack_di_*` with no repacking at all. That was checked
 * against the 7.x headers before any of this was written, not assumed.
 *
 * The `di` family is the `int32_t` index / `double` value one, which matches koblas's `IntArray` and
 * `DoubleArray` exactly. The `dl` family takes `int64_t` and would need a widening copy; if a caller ever
 * has a matrix with more than 2^31 nonzeros, that is the variant to add.
 *
 * Nothing is bundled and nothing is linked. `libumfpack` is looked up by soname, so this backend exists
 * exactly on machines that have SuiteSparse installed and is absent otherwise, with koblas's portable
 * `SparseLu` taking over — the same arrangement the dense host-OpenBLAS binding uses.
 *
 * Calls go through `invokeWithArguments` rather than `invokeExact` for the reason `HostBlasCalls` documents:
 * `invokeExact` is signature-polymorphic and Kotlin boxes its arguments instead of emitting it, after which
 * the native side reads garbage.
 */
internal object UmfpackCalls {

    /** `umfpack_di_solve`'s `sys` selector: `Ax = b`. */
    const val SYS_A = 0

    /** `umfpack_di_solve`'s `sys` selector: `Aᵀx = b`. For real matrices this is the plain transpose. */
    const val SYS_AT = 1

    /** `Control` array length (`UMFPACK_CONTROL`); passing NULL takes the defaults instead. */
    const val CONTROL = 20

    /** `Info` array length (`UMFPACK_INFO`). */
    const val INFO = 90

    /** `umfpack_di_*` success. */
    const val OK = 0

    /** `UMFPACK_WARNING_singular_matrix`: a factorization was produced, but the matrix is singular. */
    const val WARNING_SINGULAR = 1

    /** Whether `libumfpack` resolved on this machine. */
    val available: Boolean

    private val linker = Linker.nativeLinker()

    // UMFPACK reads and writes the caller's arrays; critical lets a Kotlin array be handed over pinned
    // rather than copied, which is what makes a per-solve call worth making at all.
    private val critical = Linker.Option.critical(true)

    private var symbolicHandle: MethodHandle? = null
    private var numericHandle: MethodHandle? = null
    private var solveHandle: MethodHandle? = null
    private var freeSymbolicHandle: MethodHandle? = null
    private var freeNumericHandle: MethodHandle? = null
    private var determinantHandle: MethodHandle? = null

    init {
        var resolved = false
        try {
            val lookup = openUmfpack()
            if (lookup != null) {
                // int umfpack_di_symbolic(int n_row, int n_col, const int Ap[], const int Ai[],
                //                         const double Ax[], void **Symbolic, const double Control[],
                //                         double Info[])
                symbolicHandle = bind(
                    lookup,
                    "umfpack_di_symbolic",
                    FunctionDescriptor.of(
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                    ),
                )
                // int umfpack_di_numeric(const int Ap[], const int Ai[], const double Ax[], void *Symbolic,
                //                        void **Numeric, const double Control[], double Info[])
                numericHandle = bind(
                    lookup,
                    "umfpack_di_numeric",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
                )
                // int umfpack_di_solve(int sys, const int Ap[], const int Ai[], const double Ax[],
                //                      double X[], const double B[], void *Numeric,
                //                      const double Control[], double Info[])
                solveHandle = bind(
                    lookup,
                    "umfpack_di_solve",
                    FunctionDescriptor.of(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                    ),
                )
                freeSymbolicHandle = bind(lookup, "umfpack_di_free_symbolic", FunctionDescriptor.ofVoid(ADDRESS))
                freeNumericHandle = bind(lookup, "umfpack_di_free_numeric", FunctionDescriptor.ofVoid(ADDRESS))
                // int umfpack_di_get_determinant(double *Mx, double *Ex, void *Numeric, double Info[])
                determinantHandle = bind(
                    lookup,
                    "umfpack_di_get_determinant",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
                )
                resolved = symbolicHandle != null &&
                    numericHandle != null &&
                    solveHandle != null &&
                    freeNumericHandle != null
            }
        } catch (_: Throwable) {
            // No SuiteSparse on this machine, or a version without the di family: stay unavailable so the
            // portable SparseLu keeps the seam. Throwable because a missing library surfaces as an Error.
            resolved = false
        }
        available = resolved
    }

    /**
     * Opens `libumfpack` by soname, newest first.
     *
     * Versioned sonames come first because a bare `libumfpack.so` is the development symlink and need not be
     * installed on a runtime-only machine. The dependent SuiteSparse libraries (amd, colamd, cholmod,
     * suitesparseconfig) arrive through `DT_NEEDED` rather than needing their own lookups.
     */
    private fun openUmfpack(): SymbolLookup? {
        for (soname in listOf("libumfpack.so.6", "libumfpack.so.5", "libumfpack.so", "libumfpack.dylib")) {
            @Suppress("TooGenericExceptionCaught") // a missing library is an Error, not an Exception
            try {
                return SymbolLookup.libraryLookup(soname, Arena.global())
            } catch (_: Throwable) {
                continue
            }
        }
        return null
    }

    private fun bind(lookup: SymbolLookup, name: String, descriptor: FunctionDescriptor): MethodHandle? {
        val address = lookup.find(name).orElse(null) ?: return null
        return linker.downcallHandle(address, descriptor, critical)
    }

    private fun handle(h: MethodHandle?, name: String): MethodHandle =
        requireNotNull(h) { "umfpack: $name did not resolve; UmfpackCalls.available should have been checked" }

    /** `umfpack_di_symbolic`: the pattern analysis, which does not look at [values]. */
    fun symbolic(
        n: Int,
        colPtr: MemorySegment,
        rowIdx: MemorySegment,
        values: MemorySegment,
        symbolicOut: MemorySegment,
        info: MemorySegment,
    ): Int = handle(symbolicHandle, "umfpack_di_symbolic").invokeWithArguments(
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
    ): Int = handle(numericHandle, "umfpack_di_numeric").invokeWithArguments(
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
     * iterative refinement by default, and the residual needs the original `A`. That is also why the
     * factorization has to keep its `SparseMatrix` alive.
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
    ): Int = handle(solveHandle, "umfpack_di_solve").invokeWithArguments(
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

    /** `umfpack_di_get_determinant`: mantissa into `mx`, base-10 exponent into `ex`. */
    fun determinant(mx: MemorySegment, ex: MemorySegment, numeric: MemorySegment, info: MemorySegment): Int =
        handle(determinantHandle, "umfpack_di_get_determinant")
            .invokeWithArguments(mx, ex, numeric, info) as Int

    /** `umfpack_di_free_symbolic`; the analysis is not needed once the numeric factors exist. */
    fun freeSymbolic(symbolicHolder: MemorySegment) {
        freeSymbolicHandle?.invokeWithArguments(symbolicHolder)
    }

    /** `umfpack_di_free_numeric`. Not calling this leaks whatever UMFPACK malloc'd for the factors. */
    fun freeNumeric(numericHolder: MemorySegment) {
        freeNumericHandle?.invokeWithArguments(numericHolder)
    }
}
