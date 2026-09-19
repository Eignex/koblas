package com.eignex.koblas.sparse

import com.eignex.koblas.vendor.RouteKind

/**
 * A built-in sparse Level 2 or 3 entry point, named so a caller can ask what a call of it actually executes.
 *
 * The variants a caller selects with a flag are separate entries because they execute differently: a
 * transposed CSC product reduces down a column while the untransposed one scatters it, and a triangular
 * operation with the sparse operand on the right updates whole dense columns where the left-hand form works
 * one right-hand side at a time. One name per executed shape is what keeps attribution from averaging two
 * implementations under a flag.
 */
public enum class SparseMatrixOperation(internal val entryPoint: String) {
    /** `y = alpha·A·x + beta·y`, scattering each CSC column into the destination. */
    Gemv("spgemv"),

    /** `y = alpha·Aᵀ·x + beta·y`, reducing each CSC column against the dense input. */
    GemvTransposed("spgemv-transposed"),

    /** Selected-triangle symmetric matrix-vector product. */
    Symv("spsymv"),

    /** Selected-triangle symmetric product against a dense block. */
    Symm("spsymm"),

    /** `C = alpha·op(A)·op(B) + beta·C` for a sparse `A` and a dense `B`. */
    GemmDense("spmm"),

    /** `C = alpha·op(B)·op(A) + beta·C`, the same product with the sparse operand on the right. */
    GemmDenseRight("spmm-right"),

    /** `alpha·op(A)·op(B)` for two sparse operands, into a fresh CSC result. */
    GemmSparse("spgemm"),

    /** The same product accumulated straight into a dense destination. */
    GemmSparseDense("spgemm-dense"),

    /** `C = alpha·op(A)·op(A)ᵀ + beta·C` into a dense selected triangle. */
    SyrkDense("spsyrk-dense"),

    /** The selected triangle of `op(A)·op(A)ᵀ` into a fresh CSC result. */
    SyrkSparse("spsyrk-sparse"),

    /** Sparse triangular solve for one right-hand side. */
    Trsv("sptrsv"),

    /** Sparse triangular multiply for one right-hand side. */
    Trmv("sptrmv"),

    /** Sparse triangular solve with the triangle on the left of a dense block. */
    TrsmLeft("sptrsm"),

    /** Sparse triangular solve with the triangle on the right of a dense block. */
    TrsmRight("sptrsm-right"),

    /** Sparse triangular multiply with the triangle on the left of a dense block. */
    TrmmLeft("sptrmm"),

    /** Sparse triangular multiply with the triangle on the right of a dense block. */
    TrmmRight("sptrmm-right"),

    /** Fresh CSC `alpha·op(A) + B`. */
    AddScaled("spadd"),

    /** Fresh CSC transpose. */
    Transpose("sptranspose"),
}

/**
 * What a built-in sparse Level 2 or 3 call executes, derived from the same decisions the call makes.
 *
 * The sparse counterpart of [SparseRoute] one level up: a whole sparse call is CSC scheduling written here,
 * which may hand each column to a Level 1 leaf that the engine selects. Neither half may stand for the other.
 * An engine whose Level 1 kernels are Vector API ones does not thereby execute a vectorised sparse product,
 * and reporting the engine's name would claim exactly that. [scheduling] is always this library's own portable
 * CSC code; [leaf] names the selected Level 1 implementation where one runs, and is null where the arithmetic
 * is the scheduling's own.
 *
 * Building a route allocates, so it belongs before a timed region and never inside one.
 */
public class SparseMatrixRoute internal constructor(
    /** The operation requested. */
    public val operation: SparseMatrixOperation,
    /** How the call was served. */
    public val kind: RouteKind,
    /** The component that owns traversal, structure and arithmetic order. */
    public val scheduling: String,
    /** The resolved entry point within [scheduling]. */
    public val entryPoint: String,
    /**
     * The Level 1 implementation and leaf each unit of work reaches, or null when none is called.
     *
     * A null is the honest answer for an operation whose arithmetic is written in the traversal itself, and
     * it is not the same as a scalar leaf: there is no separate kernel to name.
     */
    public val leaf: String?,
    /** What the leaf covers, or why the call is composed, or null when neither needs saying. */
    public val reason: String?,
) {
    /** The full attribution: the scheduling, plus the leaf it calls when there is one. */
    public val implementation: String get() = if (leaf == null) scheduling else "$scheduling+$leaf"

    /**
     * Whether this call is admissible as an exact measurement of the implementation named.
     *
     * A sparse call with a uniform leaf is exact. One whose columns differ in length can reach a vector kernel
     * for some and its scalar fallback for others, and is reported [RouteKind.Composed] so a benchmark does
     * not publish a mixture under either name.
     */
    public val exactlyMeasurable: Boolean get() = kind == RouteKind.Direct

    override fun toString(): String = buildString {
        append(operation.name.lowercase())
        append(' ')
        append(kind.name.lowercase())
        append(' ')
        append(implementation)
        append(' ')
        append(entryPoint)
        reason?.let { append(" (").append(it).append(')') }
    }
}
