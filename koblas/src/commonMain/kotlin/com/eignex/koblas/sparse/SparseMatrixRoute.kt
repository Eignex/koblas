package com.eignex.koblas.sparse

import com.eignex.koblas.MatrixRoute
import com.eignex.koblas.SparseMatrix
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

    /** Selected-triangle symmetric product with the sparse operand on the left of a dense block. */
    SymmLeft("spsymm"),

    /** The same product with the sparse operand on the right, where every update is a whole dense column. */
    SymmRight("spsymm-right"),

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

    /** Copying a source into an immutable prepared snapshot, which runs no arithmetic kernel. */
    Prepare("spprepare"),
}

/**
 * Facts used by expert shape diagnostics, inspected without executing arithmetic.
 *
 * The actual CSC column lengths determine indexed leaves. Destination elements, product depth, whole-column
 * update length and right-hand-side count describe the requested work; null means an extent was not supplied.
 * Transpose and triangle flags match the operation. Missing facts produce an unresolved diagnostic.
 *
 * @property matrix CSC operand whose column lengths are inspected.
 * @property alpha Operand multiplier.
 * @property beta Destination multiplier.
 * @property destinationElements Dense destination length, or null when not supplied.
 * @property depth Shared product dimension, or null when not applicable.
 * @property updateRun Length of a whole dense column update.
 * @property rightHandSides Number of dense right-hand sides.
 * @property transposeSparse Whether the sparse operand is transposed.
 * @property transposeDense Whether the dense operand is transposed.
 * @property lower Selected stored triangle.
 */
public class SparseCall(
    public val matrix: SparseMatrix,
    public val alpha: Double = 1.0,
    public val beta: Double = 1.0,
    public val destinationElements: Int? = null,
    public val depth: Int? = null,
    public val updateRun: Int = 0,
    public val rightHandSides: Int = 0,
    public val transposeSparse: Boolean = false,
    public val transposeDense: Boolean = false,
    public val lower: Boolean = true,
) {
    init {
        require(destinationElements == null || destinationElements >= 0) { "negative destination element count" }
        require(depth == null || depth >= 0) { "negative product depth" }
        require(updateRun >= 0) { "negative update run length" }
        require(rightHandSides >= 0) { "negative right-hand side count" }
    }
}

/** Shape diagnostics. Components name reached bodies; composed routes may also execute scalar traversal. */
public class SparseMatrixRoute internal constructor(
    /** The operation requested. */
    public val operation: SparseMatrixOperation,
    /** How the call was served. */
    override val kind: RouteKind,
    /** The component that owns traversal, structure and arithmetic order. */
    override val scheduling: String,
    /** The resolved entry point within [scheduling]. */
    override val entryPoint: String,
    /** Every Level 1 implementation and leaf the call reaches, in call order. */
    override val components: List<String>,
    /**
     * Right-hand sides this call visits per walk of a sparse column, or zero where it groups none.
     *
     * The local geometry the backend recommended for the layout this call will have, within the ceiling the
     * sparse scheduling keeps for a staged copy. It is not a lane count: a group of four right-hand sides
     * and four lanes are different numbers that agree on some machines.
     */
    override val executionGroup: Int = 0,
    /**
     * Right-hand sides in the last group, where that group is shorter than [executionGroup], or zero where
     * every group is full.
     *
     * Named separately because a short last group may reach a different body from the full ones, which is
     * what makes such a call a composition.
     */
    public val executionTail: Int = 0,
    /**
     * Whether the facts this route was asked about settle what the call runs.
     *
     * False where a decision depends on something a [SparseCall] does not carry, such as the second operand
     * a prepared transposed sparse product orients on, or a right-hand side count a caller left out. Such a
     * route still names whatever is known, the destination scaling included.
     * A report has to publish this rather than infer it from an empty component list, since a call can be
     * undecided and still scale a destination.
     */
    public val resolved: Boolean = true,
) : MatrixRoute {
    /** Logical grouping, with a shorter last group when present. */
    public val groupSuffix: String
        get() = when {
            executionGroup <= 0 -> ""
            executionTail > 0 -> "@$executionGroup+$executionTail"
            else -> "@$executionGroup"
        }

    override fun toString(): String = buildString {
        append(operation.name.lowercase())
        append(' ')
        append(kind.name.lowercase())
        append(' ')
        append(implementation)
        append(' ')
        append(entryPoint)
        append(groupSuffix)
        if (!resolved) append(" unresolved")
    }
}
