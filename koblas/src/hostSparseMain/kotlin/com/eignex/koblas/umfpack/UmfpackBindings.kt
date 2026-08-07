@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.umfpack

import kotlinx.cinterop.CFunction
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.COpaquePointerVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.invoke
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import platform.posix.RTLD_NOW
import platform.posix.dlopen
import platform.posix.dlsym

private typealias Dp = CPointer<DoubleVar>?
private typealias Ip = CPointer<IntVar>?
private typealias Hp = CPointer<COpaquePointerVar>?

/** `umfpack_di_solve`'s `sys` selector: `Ax = b`. */
internal const val SYS_A = 0

/** `umfpack_di_solve`'s `sys` selector: `Aᵀx = b`. For real matrices this is the plain transpose. */
internal const val SYS_AT = 1

/** `Info` array length (`UMFPACK_INFO`). */
internal const val INFO = 90

/** `Control` array length (`UMFPACK_CONTROL`). */
private const val CONTROL = 20

/** `Control` index of the iterative-refinement step count (`UMFPACK_IRSTEP`); UMFPACK defaults it to 2. */
private const val IRSTEP = 7

/** `umfpack_di_*` success. */
internal const val OK = 0

/** `UMFPACK_WARNING_singular_matrix`: a factorization was produced, but the matrix is singular. */
internal const val WARNING_SINGULAR = 1

/** `Info[UMFPACK_LNZ]`: nonzeros in `L`, diagonal included. */
internal const val INFO_LNZ = 43

/** `Info[UMFPACK_UNZ]`: nonzeros in `U`, diagonal included. */
internal const val INFO_UNZ = 44

/**
 * The UMFPACK subset koblas dispatches to on the native targets, resolved with `dlopen`/`dlsym`.
 *
 * The same subset the JVM binding covers, and the same reasoning: SuiteSparse's input format *is* koblas's,
 * so a `SparseMatrix` crosses to `umfpack_di_*` with no repacking. The signatures come from the installed
 * `umfpack.h`. Prototypes are LP64 with `int` indices, which is the `di` family; `dl` would need widening
 * copies of every index array.
 *
 * Resolved rather than linked, exactly as the CBLAS binding beside it is: a binary carrying this backend
 * still starts on a host without SuiteSparse, and falls back to koblas's portable `SparseLu`.
 */
@Suppress("MagicNumber") // the prototypes' arities are ABI facts
internal class UmfpackFunctions(private val lib: COpaquePointer) {
    private fun required(name: String): COpaquePointer = dlsym(lib, name)
        ?: error("libumfpack is present but lacks $name")

    // int umfpack_di_symbolic(int n_row, int n_col, const int Ap[], const int Ai[], const double Ax[],
    //                         void **Symbolic, const double Control[], double Info[])
    val symbolic = required("umfpack_di_symbolic")
        .reinterpret<CFunction<(Int, Int, Ip, Ip, Dp, Hp, Dp, Dp) -> Int>>()

    // int umfpack_di_numeric(const int Ap[], const int Ai[], const double Ax[], void *Symbolic,
    //                        void **Numeric, const double Control[], double Info[])
    val numeric = required("umfpack_di_numeric")
        .reinterpret<CFunction<(Ip, Ip, Dp, COpaquePointer?, Hp, Dp, Dp) -> Int>>()

    // int umfpack_di_solve(int sys, const int Ap[], const int Ai[], const double Ax[], double X[],
    //                      const double B[], void *Numeric, const double Control[], double Info[])
    val solve = required("umfpack_di_solve")
        .reinterpret<CFunction<(Int, Ip, Ip, Dp, Dp, Dp, COpaquePointer?, Dp, Dp) -> Int>>()

    val freeSymbolic = required("umfpack_di_free_symbolic")
        .reinterpret<CFunction<(Hp) -> Unit>>()

    val freeNumeric = required("umfpack_di_free_numeric")
        .reinterpret<CFunction<(Hp) -> Unit>>()

    // int umfpack_di_get_determinant(double *Mx, double *Ex, void *Numeric, double Info[])
    val determinant = required("umfpack_di_get_determinant")
        .reinterpret<CFunction<(Dp, Dp, COpaquePointer?, Dp) -> Int>>()

    // void umfpack_di_defaults(double Control[UMFPACK_CONTROL])
    val defaults = required("umfpack_di_defaults")
        .reinterpret<CFunction<(Dp) -> Unit>>()
}

/**
 * Locates `libumfpack` once, and holds the `Control` array every solve passes.
 *
 * [functions] is null when there is no usable UMFPACK, in which case the sparse half stays portable. A
 * library that opens but lacks a required symbol counts as absent rather than as a broken installation to
 * crash on, which is what [UmfpackFunctions.required] throwing gets turned into here.
 */
internal object UmfpackLoader {
    private val handle: COpaquePointer? = open(
        "libumfpack.so.6", // versioned sonames first: a bare .so is the development symlink
        "libumfpack.so.5",
        "libumfpack.so",
        "libumfpack.dylib",
        "/opt/homebrew/opt/suite-sparse/lib/libumfpack.dylib", // Homebrew is keg-only
        "/usr/local/opt/suite-sparse/lib/libumfpack.dylib",
    )

    val functions: UmfpackFunctions? = handle?.let { lib ->
        try {
            UmfpackFunctions(lib)
        } catch (_: IllegalStateException) { // a required symbol is missing: treat as not installed
            null
        }
    }

    /**
     * UMFPACK's own defaults with iterative refinement turned off — see the JVM `UmfpackCalls.solveControl`
     * for the measurements behind that.
     *
     * Filled by `umfpack_di_defaults` rather than left zeroed, because a zeroed `Control` overrides the pivot
     * tolerance and the dense-column heuristics with zeros instead of meaning "no opinion". Kept as a Kotlin
     * array and pinned per call: it is 20 doubles, and pinning one more array beside the five a solve already
     * pins costs nothing measurable against a sparse solve.
     */
    val control: DoubleArray? = functions?.let { f ->
        val values = DoubleArray(CONTROL)
        values.usePinned { f.defaults(it.addressOf(0)) }
        values[IRSTEP] = 0.0
        values
    }

    /** Whether UMFPACK resolved. The native counterpart of the JVM's `UmfpackCalls.available`. */
    val available: Boolean get() = functions != null

    private fun open(vararg names: String): COpaquePointer? {
        for (name in names) {
            val opened = dlopen(name, RTLD_NOW) ?: continue
            // A library that opens without the di family cannot serve, and must not shadow a later soname.
            if (dlsym(opened, "umfpack_di_symbolic") != null) return opened
        }
        return null
    }
}
