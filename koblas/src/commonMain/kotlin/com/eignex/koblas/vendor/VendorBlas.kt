@file:Suppress("TooManyFunctions") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.dense.MatrixStructure

/**
 * One vendor BLAS library, bound and callable.
 *
 * Operands are the storage types themselves: a [DenseMatrix] is contiguous column-major with a leading
 * dimension equal to its row count, and a [DenseVector] is a pointer, a length and an increment. How a call
 * reads an operand, its transpose and its stored triangle, travels beside it as a flag rather than inside it,
 * because those describe the call rather than the storage.
 *
 * The ABI is fixed: double precision, 32-bit BLAS integers matching Kotlin [Int], and the unsuffixed CBLAS
 * symbols that carry it. A library whose integers are 64 bits is a different ABI and is rejected at load
 * rather than producing wrong answers on large inputs.
 *
 * Instances are immutable and safe to share. Scratch used for staging is exclusive to one call, so concurrent
 * calls on one instance do not interfere.
 *
 * Ownership is the same for every call here, and is what an owning-container seam above this one has to
 * preserve. A window borrows its array and does not extend its life: the caller keeps it reachable for the
 * call, nothing is retained afterwards, and no foreign pointer outlives the call that made it. An operand
 * copied into native memory lives in that call's arena and is freed with it, because caching a copy of a
 * mutable array across calls would be a correctness bug rather than an optimization. A call writes only what
 * its destination window addresses, plus whatever padding that window's own leading dimension already spanned,
 * which is carried across a transfer unchanged; storage outside the window is left as the caller left it.
 * Distinct windows may share one array where an operation says so, as the triangular solves do for their
 * in-place destination, and where an operation forbids overlap the call requires it rather than discovering it
 * late. Concurrent readers are fine; a write overlapping another call's access is the caller's to order.
 * Structure is declared rather than inferred, so an entry the structure says is absent is never read and an
 * implicit unit diagonal may hold anything.
 */
public interface VendorBlas {
    /** Which library this is. */
    public val vendor: Vendor

    /** The resolved library file, which is the evidence that a named vendor is the one that ran. */
    public val libraryPath: String

    /** The library's own version string, or a note that it does not report one. */
    public val version: String

    /**
     * Whether the one compute thread every call runs on was confirmed against this library.
     *
     * A library that reported more than one thread after being held to one never became a binding at all, so
     * this distinguishes a library that confirmed the requirement from one that cannot be asked. It is evidence
     * for a report to carry, not a setting: there is nothing to change and no thread count to pass.
     */
    public val threadEvidence: ThreadEvidence

    /**
     * The operations this library exports directly.
     *
     * An operation being listed does not authorize a benchmark case on its own: the same operation can still
     * decline a particular window. [routeOf] answers for a concrete call; this answers for the layer.
     */
    public val directlyImplemented: Set<VendorOperation>

    /**
     * What a call to [operation] over these operands would do, each list in the order the operation names them.
     *
     * Both lists matter, because a route describes one concrete call and not an operation in general. An
     * operation asked about with no operands can only be answered in general terms, and the answer would be
     * wrong for a call that returns without reaching BLAS: a dot over an empty window does no vendor work, and
     * describing it as a direct vendor call is the attribution error this contract exists to prevent. Passing
     * the operands the call will use is what makes the answer specific to it.
     *
     * Allocates, so callers resolve it before timing rather than inside a timed call.
     */
    public fun routeOf(
        operation: VendorOperation,
        matrices: List<DenseMatrix> = emptyList(),
        vectors: List<DenseVector> = emptyList(),
    ): CallRoute

    /** `xᵀ · y`. */
    public fun dot(x: DenseVector, y: DenseVector): Double

    /** `‖x‖₂`, computed by the vendor's own overflow-avoiding method. */
    public fun nrm2(x: DenseVector): Double

    /** `Σ|xᵢ|`. */
    public fun asum(x: DenseVector): Double

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
    public fun iamax(x: DenseVector): Int

    /** `y = alpha · x + y`. */
    public fun axpy(alpha: Double, x: DenseVector, y: DenseVector)

    /**
     * `x = alpha · x`.
     *
     * A negatively strided window is passed with the increment's magnitude. BLAS defines `scal` to return
     * without doing anything for a non-positive increment, so keeping the sign would silently leave the window
     * unscaled; scaling is order-independent, so walking the same entries forwards gives the same result.
     */
    public fun scal(alpha: Double, x: DenseVector)

    /** `y = x`. */
    public fun copy(x: DenseVector, y: DenseVector)

    /** Exchanges the entries of [x] and [y]. */
    public fun swap(x: DenseVector, y: DenseVector)

    /** Applies the plane rotation given by [c] and [s] to [x] and [y]. */
    public fun rot(x: DenseVector, y: DenseVector, c: Double, s: Double)

    /**
     * Generates the modified Givens transformation eliminating the second component, and the updated scaling
     * state that goes with it (BLAS `drotmg`).
     *
     * The library owns the branch conditions, so a vendor whose reference translation differs at a boundary,
     * a negative `d1` or a zero `d2 * y1`, answers differently from the portable one. Four scalars in and five
     * out is far less work than the call that carries them, so this exists to be compared rather than because
     * a downcall is the quick way to compute it.
     */
    public fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens

    /**
     * Applies a modified Givens [transformation] to `x` and `y` (BLAS `drotm`).
     *
     * An identity transformation, whose flag is `-2.0`, reads neither operand, which is the library's rule and
     * not one imposed here.
     */
    public fun rotm(x: DenseVector, y: DenseVector, transformation: ModifiedGivens)

    /** `y = alpha · op(A) · x + beta · y`, transposing [a] when [transposeA]. */
    public fun gemv(alpha: Double, a: DenseMatrix, transposeA: Boolean, x: DenseVector, beta: Double, y: DenseVector)

    /** `y = alpha · A · x + beta · y` for symmetric [a], whose [structure] selects the stored triangle. */
    public fun symv(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    )

    /** `A = alpha · x · yᵀ + A`. */
    public fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix)

    /** `A = alpha · x · xᵀ + A` in the triangle [structure] selects. */
    public fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, structure: MatrixStructure)

    /** `A = alpha · (x · yᵀ + y · xᵀ) + A` in the triangle [structure] selects. */
    public fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, structure: MatrixStructure)

    /** Solves `op(A) · x = b` in place; [structure] carries the triangle and unit diagonal. */
    public fun trsv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector)

    /** `x = op(A) · x`; [structure] carries the triangle and unit diagonal. */
    public fun trmv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector)

    /** `C = alpha · op(A) · op(B) + beta · C`, transposing each operand as its flag says. */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    public fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    )

    /** `C = alpha · A · B + beta · C` for symmetric [a], or `C = alpha · B · A + beta · C` when [rightSide]. */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    public fun symm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        rightSide: Boolean = false,
    )

    /** `C = alpha · op(A) · op(A)ᵀ + beta · C` in the triangle [structure] selects. */
    @Suppress("LongParameterList") // the BLAS dsyrk signature
    public fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    )

    /** `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` in the triangle [structure] selects. */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    public fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    )

    /** `B = alpha · op(A) · B`, or `B = alpha · B · op(A)` when [rightSide], for triangular [a]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    public fun trmm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean = false,
    )

    /** Solves `op(A) · X = alpha · B`, or `X · op(A) = alpha · B` when [rightSide], in place over [b]. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean = false,
    )

    /**
     * `C = alpha · op(A) · op(B) + beta · C` in the triangle [c]'s structure selects, leaving the other
     * triangle unread and unwritten.
     *
     * Direct where the vendor exports it and composed from [gemm] plus a triangle copy where it does not, so
     * the route of a call says which one ran rather than leaving the name to imply the first.
     */
    @Suppress("LongParameterList") // the BLAS gemmt signature
    public fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    )
}

/** Both bindings reject a mismatched pair of vector operands the same way, and with the same message. */
internal fun requireSameLength(x: DenseVector, y: DenseVector, what: String) {
    require(x.size == y.size) { "$what: vector sizes differ" }
}

/**
 * A square operand with a stored triangle and a stored diagonal.
 *
 * None of the routines that take one carry a `diag` flag, so there is no way to tell the vendor that a
 * diagonal is implied. A window that says its diagonal is implicit is therefore rejected rather than served by
 * a call that would read or write it anyway; the triangular routines, which do carry the flag, take
 * [requireTriangular] instead.
 */
internal fun requireStructured(a: DenseMatrix, structure: MatrixStructure, what: String) {
    val stored = structure != MatrixStructure.General &&
        structure != MatrixStructure.UnitLower &&
        structure != MatrixStructure.UnitUpper
    require(stored) { "$what requires a stored triangle with a stored diagonal" }
    require(a.rows == a.cols) { "$what requires a square matrix" }
}

/** A triangular operand, stored or with an implicit unit diagonal. */
internal fun requireTriangular(a: DenseMatrix, structure: MatrixStructure, what: String) {
    val triangular = structure == MatrixStructure.TriangularLower ||
        structure == MatrixStructure.TriangularUpper ||
        structure == MatrixStructure.UnitLower ||
        structure == MatrixStructure.UnitUpper
    require(triangular) { "$what requires a triangular matrix" }
    require(a.rows == a.cols) { "$what requires a square matrix" }
}

/**
 * [VendorBlas.gemmt] assembled from a full [VendorBlas.gemm] plus a copy of the selected triangle, for a
 * vendor that does not export `cblas_dgemmt`.
 *
 * The composition keeps the contract the direct call has: the triangle [c]'s structure does not select is
 * neither read nor written, an implicit unit diagonal is left alone, and a zero [beta] does not read the
 * destination. Which triangle that is comes from [uploFor], the same rule the direct call passes to the
 * vendor, so the two paths cannot disagree about where the result lands. It costs a full product either way,
 * which is why the route of the call says which path ran.
 */
@Suppress("LongParameterList") // the BLAS gemmt signature
internal fun VendorBlas.composeGemmt(
    alpha: Double,
    a: DenseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    structure: MatrixStructure,
) {
    val order = c.rows
    val product = DenseMatrix.zero(order, order)
    gemm(alpha, a, transposeA, b, transposeB, 0.0, product)
    val lower = uploFor(structure) == Cblas.LOWER
    for (column in 0 until order) {
        val from = if (lower) column else 0
        val until = if (lower) order else column + 1
        for (row in from until until) {
            val index = row + column * order
            val previous = if (beta == 0.0) 0.0 else beta * c.data[index]
            c.data[index] = previous + product.data[index]
        }
    }
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
