package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.VectorRoute
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
 * The facts about one concrete sparse matrix call that decide what executes.
 *
 * A route needs the call, not the operation: which Level 1 implementation a column reaches depends on how long
 * that column is, whether any kernel runs at all depends on the scalars and the extents, and how a dense
 * operand is walked depends on its shape. Passing a single representative column length instead would name a
 * leaf the call reaches only sometimes, which is the thing route reporting exists to prevent.
 *
 * The two extents are nullable rather than zero-defaulted because an operation that writes no dense
 * destination and one whose dense destination is empty are different calls, and only one of them does nothing.
 *
 * @property matrix the sparse operand, whose actual column lengths are inspected.
 * @property alpha the multiplier the call will use.
 * @property beta the destination multiplier, or 1.0 for an operation that has none.
 * @property destinationElements elements of the dense destination the call writes, or null when it writes
 *   none. Zero is a destination with no elements, which is a call with nothing to do.
 * @property depth the inner extent of a product, or null when the operation has none. Zero is a product over
 *   nothing, which is also a call with nothing to do.
 * @property updateRun contiguous dense elements one update covers, for an operation whose unit of work is a
 *   whole dense column, or 0 when it has no such unit.
 * @property rightHandSides dense right-hand sides an operation with a dense block carries, or 0 when it has
 *   none. How many are visited together, and whether they are staged adjacent first, follow from this and
 *   from what the operand holds.
 * @property transposeSparse whether the sparse operand is transposed, which decides whether a product
 *   reduces a column into one output row or spreads one index across the rows it stores.
 * @property transposeDense whether the dense operand is transposed, which decides whether its right-hand
 *   sides are already adjacent.
 * @property lower which triangle a symmetric or triangular operation selects, which decides how much of
 *   each column it hands to a panel and whether it hands over anything at all. An operation with no
 *   selected triangle ignores it.
 *
 * A unit diagonal is deliberately not among these. It removes a division from a pivot and changes no body
 * any panel reaches, because the run a panel is handed is the strictly triangular part of a column either
 * way.
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

/**
 * What a built-in sparse Level 2 or 3 call executes, derived from the same decisions the call makes.
 *
 * The sparse counterpart of [VectorRoute] one level up: a whole sparse call is CSC scheduling written in this
 * library, which may hand individual units of work to Level 1 kernels the engine selected. Neither half may
 * stand for the other. An engine whose Level 1 kernels are Vector API ones does not thereby execute a
 * vectorised sparse product, and reporting the engine's name would claim exactly that.
 *
 * [components] lists the Level 1 implementations and leaves the call reaches, in the order it reaches them,
 * including the destination scaling that a nonzero and non-unit multiplier performs. An empty list means the
 * arithmetic is the traversal's own and no kernel is called, which is not the same as a scalar kernel.
 *
 * What the list promises depends on [kind]. A [RouteKind.Direct] route names what runs: every unit of work
 * reaches every component. A [RouteKind.Composed] route names what may run: either the units differ, because
 * a matrix's columns straddle a kernel's crossover, or the traversal decides per unit whether the named kernel
 * is called at all, because a stored zero coefficient or an unselected triangle position skips its update.
 * Such a route therefore does not assert that a vector kernel executed, only that it is the one this call can
 * reach, and [reason] says which of the two it is. [RouteKind.NoWork] is a call whose own contract stops
 * before the arithmetic, where nothing but the destination scaling runs.
 *
 * Building a route inspects the operand, so it belongs before a timed region and never inside one.
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
    /** Every Level 1 implementation and leaf the call reaches, in call order. */
    public val components: List<String>,
    /** What the components cover, what the traversal keeps for itself, or why the call is composed. */
    public val reason: String?,
    /**
     * Right-hand sides this call visits per walk of a sparse column, or zero where it groups none.
     *
     * The local geometry the backend recommended for the layout this call will have, within the ceiling the
     * sparse scheduling keeps for a staged copy. It is not a lane count: a group of four right-hand sides
     * and four lanes are different numbers that agree on some machines.
     */
    public val executionGroup: Int = 0,
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
     * route still names whatever is known, the destination scaling included, and [reason] says what is not.
     * A report has to publish this rather than infer it from an empty component list, since a call can be
     * undecided and still scale a destination.
     */
    public val resolved: Boolean = true,
) {
    /** The full attribution: the scheduling, plus every component it calls. */
    public val implementation: String
        get() = if (components.isEmpty()) scheduling else components.joinToString("+", prefix = "$scheduling+")

    /**
     * Whether this call is admissible as an exact measurement of the implementations named.
     *
     * A call whose units of work do not all reach the same implementation is reported [RouteKind.Composed] so
     * a benchmark publishes it as a composition rather than under either name on its own.
     */
    public val exactlyMeasurable: Boolean get() = kind == RouteKind.Direct

    /** The grouping as a report writes it: the width, and the last group where one is short. */
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
        reason?.let { append(" (").append(it).append(')') }
    }
}
