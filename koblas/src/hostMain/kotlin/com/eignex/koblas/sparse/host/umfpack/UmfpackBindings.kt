@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.internal.host.openNativeLibrary
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
import platform.posix.dlsym

private typealias Dp = CPointer<DoubleVar>?
private typealias Ip = CPointer<IntVar>?
private typealias Hp = CPointer<COpaquePointerVar>?

/** Prototypes are LP64 with int indices, the di family; the dl family would need widened index copies. */
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

    // void umfpack_di_defaults(double Control[UMFPACK_CONTROL])
    val defaults = required("umfpack_di_defaults")
        .reinterpret<CFunction<(Dp) -> Unit>>()
}

internal class UmfpackLoader(private val config: UmfpackConfig) {
    private val handle: COpaquePointer? = openNativeLibrary(
        config.libraryPath?.let(::listOf) ?: UMFPACK_SONAMES,
        KEY_SYMBOL,
    )

    val functions: UmfpackFunctions? = handle?.let { lib ->
        try {
            UmfpackFunctions(lib)
        } catch (_: IllegalStateException) { // a required symbol is missing, treat as not installed
            null
        }
    }

    /**
     * UMFPACK's defaults with this instance's requested policy. Filled by umfpack_di_defaults rather than
     * left zeroed, since a zeroed Control overrides the remaining heuristics with zeros.
     */
    private fun control(scaling: UmfpackScaling): DoubleArray? = functions?.let { f ->
        val values = DoubleArray(CONTROL)
        values.usePinned { f.defaults(it.addressOf(0)) }
        values[IRSTEP] = config.iterativeRefinementSteps.toDouble()
        values[PIVOT_TOLERANCE] = config.pivotTolerance
        values[SCALE] = scaling.nativeValue
        values
    }

    private val scaledControl: DoubleArray? by lazy { control(config.scaling) }
    private val unscaledControl: DoubleArray? by lazy { control(UmfpackScaling.NONE) }

    /** The Control array whose scaling agrees with one [F64SparseLu.factor] request. */
    fun control(equilibrate: Boolean): DoubleArray? = if (equilibrate) scaledControl else unscaledControl

    /** The refinement steps a solve will run, or null when no Control array could be built. */
    val refinementSteps: Double? get() = scaledControl?.get(IRSTEP)

    /** The pivot tolerance, for the test that the array holds UMFPACK's defaults and not zeros. */
    val pivotTolerance: Double? get() = scaledControl?.get(PIVOT_TOLERANCE)

    val available: Boolean get() = functions != null
}

private val UmfpackScaling.nativeValue: Double
    get() = when (this) {
        UmfpackScaling.NONE -> 0.0
        UmfpackScaling.SUM -> 1.0
        UmfpackScaling.MAX -> 2.0
    }
