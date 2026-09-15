@file:Suppress("TooManyFunctions") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow

/**
 * One vendor BLAS library, bound and callable.
 *
 * Operands are [MatrixWindow] and [VectorWindow], so an offset submatrix reaches BLAS as a leading dimension
 * rather than as a copy whenever its strides allow it. When they do not, the call stages a packed copy and
 * says so in its route; it never silently becomes a different algorithm.
 *
 * The ABI is fixed: double precision, 32-bit BLAS integers matching Kotlin [Int], and the unsuffixed CBLAS
 * symbols that carry it. A library whose integers are 64 bits is a different ABI and is rejected at load
 * rather than producing wrong answers on large inputs.
 *
 * Instances are immutable and safe to share. Scratch used for staging is exclusive to one call, so concurrent
 * calls on one instance do not interfere.
 */
public interface VendorBlas {
    /** Which library this is. */
    public val vendor: Vendor

    /** The resolved library file, which is the evidence that a named vendor is the one that ran. */
    public val libraryPath: String

    /** The library's own version string, or a note that it does not report one. */
    public val version: String

    /**
     * The operations this library exports directly.
     *
     * An operation being listed does not authorize a benchmark case on its own: the same operation can still
     * decline a particular window. [routeOf] answers for a concrete call; this answers for the layer.
     */
    public val directlyImplemented: Set<VendorOperation>

    /**
     * What a call to [operation] over [matrices] would do, with the matrix operands given in the order the
     * operation names them. Level 1 operations have none.
     *
     * Allocates, so callers resolve it before timing rather than inside a timed call.
     */
    public fun routeOf(operation: VendorOperation, matrices: List<MatrixWindow> = emptyList()): CallRoute

    /** `xᵀ · y`. */
    public fun dot(x: VectorWindow, y: VectorWindow): Double

    /** `‖x‖₂`, computed by the vendor's own overflow-avoiding method. */
    public fun nrm2(x: VectorWindow): Double

    /** `Σ|xᵢ|`. */
    public fun asum(x: VectorWindow): Double

    /**
     * The index of the entry of largest magnitude, or 0 for an empty vector.
     *
     * BLAS defines its three reductions to return zero for a non-positive increment rather than to walk the
     * vector backwards, so a negatively strided window is passed with the increment's magnitude and reaches
     * the same entries in the opposite order. For [nrm2] and [asum] the order does not matter. For this one it
     * decides ties: among entries of equal largest magnitude a forward window reports the first and a backward
     * one reports the last. That is a real difference from a single-pass forward scan, and it is documented
     * rather than papered over because no vendor flag expresses the forward-scan choice.
     */
    public fun iamax(x: VectorWindow): Int

    /** `y = alpha · x + y`. */
    public fun axpy(alpha: Double, x: VectorWindow, y: VectorWindow)

    /** `x = alpha · x`. */
    public fun scal(alpha: Double, x: VectorWindow)

    /** `y = x`. */
    public fun copy(x: VectorWindow, y: VectorWindow)

    /** Exchanges the entries of [x] and [y]. */
    public fun swap(x: VectorWindow, y: VectorWindow)

    /** Applies the plane rotation given by [c] and [s] to [x] and [y]. */
    public fun rot(x: VectorWindow, y: VectorWindow, c: Double, s: Double)

    /** `y = alpha · op(A) · x + beta · y`, with the transpose taken from [a]. */
    public fun gemv(alpha: Double, a: MatrixWindow, x: VectorWindow, beta: Double, y: VectorWindow)

    /** `y = alpha · A · x + beta · y` for symmetric [a], whose structure selects the stored triangle. */
    public fun symv(alpha: Double, a: MatrixWindow, x: VectorWindow, beta: Double, y: VectorWindow)

    /** `A = alpha · x · yᵀ + A`. */
    public fun ger(alpha: Double, x: VectorWindow, y: VectorWindow, a: MatrixWindow)

    /** `A = alpha · x · xᵀ + A` in the triangle [a]'s structure selects. */
    public fun syr(alpha: Double, x: VectorWindow, a: MatrixWindow)

    /** `A = alpha · (x · yᵀ + y · xᵀ) + A` in the triangle [a]'s structure selects. */
    public fun syr2(alpha: Double, x: VectorWindow, y: VectorWindow, a: MatrixWindow)

    /** Solves `op(A) · x = b` in place, with triangle, transpose and unit diagonal taken from [a]. */
    public fun trsv(a: MatrixWindow, x: VectorWindow)

    /** `x = op(A) · x`, with triangle, transpose and unit diagonal taken from [a]. */
    public fun trmv(a: MatrixWindow, x: VectorWindow)

    /** `C = alpha · op(A) · op(B) + beta · C`, with each transpose taken from its own window. */
    public fun gemm(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow)

    /** `C = alpha · A · B + beta · C` for symmetric [a], or `C = alpha · B · A + beta · C` when [rightSide]. */
    public fun symm(
        alpha: Double,
        a: MatrixWindow,
        b: MatrixWindow,
        beta: Double,
        c: MatrixWindow,
        rightSide: Boolean = false,
    )

    /** `C = alpha · op(A) · op(A)ᵀ + beta · C` in the triangle [c]'s structure selects. */
    public fun syrk(alpha: Double, a: MatrixWindow, beta: Double, c: MatrixWindow)

    /** `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` in the triangle [c] selects. */
    public fun syr2k(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow)

    /** `B = alpha · op(A) · B`, or `B = alpha · B · op(A)` when [rightSide], for triangular [a]. */
    public fun trmm(alpha: Double, a: MatrixWindow, b: MatrixWindow, rightSide: Boolean = false)

    /** Solves `op(A) · X = alpha · B`, or `X · op(A) = alpha · B` when [rightSide], in place over [b]. */
    public fun trsm(alpha: Double, a: MatrixWindow, b: MatrixWindow, rightSide: Boolean = false)

    /**
     * `C = alpha · op(A) · op(B) + beta · C` in the triangle [c]'s structure selects, leaving the other
     * triangle unread and unwritten.
     *
     * Direct where the vendor exports it and composed from [gemm] plus a triangle copy where it does not, so
     * the route of a call says which one ran rather than leaving the name to imply the first.
     */
    public fun gemmt(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow)
}

/**
 * Opens the preferred available vendor for this host, or null when none is installed.
 *
 * Resolution happens once. Containers, Level 1 and the generic primitives do not go through this and keep
 * working on a host where it returns null; only the accelerator-dependent calls fail, and they fail clearly.
 *
 * [only] restricts the search to one vendor, which is how a test or an exact benchmark arm reaches a specific
 * library without changing anything process-global. It also reaches [Vendor.OpenBlas], which [Vendor.select]
 * never returns.
 */
public expect fun openVendorBlas(only: Vendor? = null): VendorBlas?
