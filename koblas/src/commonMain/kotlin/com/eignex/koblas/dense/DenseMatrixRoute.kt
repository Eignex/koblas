package com.eignex.koblas.dense

import com.eignex.koblas.vendor.RouteKind

/**
 * A built-in dense Level 2 or 3 entry point, named so a caller can ask what a call of it actually executes.
 *
 * The variants a caller selects with a flag are separate entries where they execute differently: a
 * transposed matrix-vector product reduces down each stored column while the untransposed one accumulates
 * those columns into the destination, and the two reach different panels. One name per executed shape is
 * what keeps attribution from averaging two implementations under a flag.
 */
public enum class DenseMatrixOperation(internal val entryPoint: String) {
    /** `y = alpha·A·x + beta·y`, accumulating each stored column into the destination. */
    Gemv("gemv"),

    /** `y = alpha·Aᵀ·x + beta·y`, reducing each stored column against the input. */
    GemvTransposed("gemv-transposed"),

    /** Selected-triangle symmetric matrix-vector product. */
    Symv("symv"),

    /** `A += alpha·x·yᵀ`. */
    Ger("ger"),

    /** `A += alpha·x·xᵀ` in the selected triangle. */
    Syr("syr"),

    /** `A += alpha·(x·yᵀ + y·xᵀ)` in the selected triangle. */
    Syr2("syr2"),

    /** Triangular matrix-vector product, accumulating each stored column into the destination. */
    Trmv("trmv"),

    /** Triangular matrix-vector product, reducing each stored column against the input. */
    TrmvTransposed("trmv-transposed"),

    /** Triangular solve, substituting forward or backward over the stored columns. */
    Trsv("trsv"),

    /** Triangular solve whose substitution reduces each stored column against the finished entries. */
    TrsvTransposed("trsv-transposed"),

    /** `C = alpha·op(A)·op(B) + beta·C`. */
    Gemm("gemm"),

    /** The same product written into one triangle of a square destination. */
    Gemmt("gemmt"),

    /** Symmetric product against a dense block, on either side. */
    Symm("symm"),

    /** `C = alpha·op(A)·op(A)ᵀ + beta·C` in the selected triangle. */
    Syrk("syrk"),

    /** `C = alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + beta·C` in the selected triangle. */
    Syr2k("syr2k"),

    /** Triangular matrix multiply against a dense block. */
    Trmm("trmm"),

    /** Triangular solve against a dense block. */
    Trsm("trsm"),
}

/**
 * The facts about one concrete dense matrix call that decide what executes.
 *
 * A route needs the call, not the operation: which panel implementation a window reaches depends on how long
 * that window is, whether any kernel runs at all depends on the scalars and the extents, and a triangular
 * traversal's windows are as long as the triangle is wide at that column rather than all the same. Passing a
 * single representative length instead would name an implementation the call reaches only sometimes, which is
 * the thing route reporting exists to prevent.
 *
 * @property rows the stored rows of the operand a panel walks, which is the length of one of its columns.
 * @property columns the stored columns of that operand.
 * @property alpha the multiplier the call will use.
 * @property beta the destination multiplier, or 1.0 for an operation that has none.
 * @property contiguous whether the vector the panel shares across its columns is adjacent in memory. A
 *   strided operand is scalar work at any width, so it is part of the question.
 * @property depth the inner extent of a product, or null when the operation has none. Zero is a product over
 *   nothing, which is a call with nothing to do however large its operands are, so it is nullable rather than
 *   zero-defaulted.
 * @property lower which triangle a selected-triangle operation reads, ignored by the rest. Part of the
 *   question because the windows a triangular traversal cuts are not the same on the two sides once a group
 *   covers more than one column.
 */
public class DenseCall(
    public val rows: Int,
    public val columns: Int,
    public val alpha: Double = 1.0,
    public val beta: Double = 1.0,
    public val contiguous: Boolean = true,
    public val depth: Int? = null,
    public val lower: Boolean = true,
) {
    init {
        require(rows >= 0) { "negative row count" }
        require(columns >= 0) { "negative column count" }
        require(depth == null || depth >= 0) { "negative product depth" }
    }
}

/**
 * What a built-in dense Level 2 or 3 call executes, derived from the same decisions the call makes.
 *
 * A whole dense call is shared scheduling written in this library, which hands units of work to the panel
 * and Level 1 kernels the engine selected. Neither half may stand for the other: an engine whose Level 1
 * kernels are Vector API ones does not thereby execute a vectorised matrix operation, and reporting the
 * engine's name would claim exactly that.
 *
 * [components] lists the implementations and leaves the call reaches, in the order it reaches them,
 * including the destination scaling that a nonzero and non-unit multiplier performs. An empty list means the
 * arithmetic is the traversal's own and no kernel is called, which is not the same as a scalar kernel.
 *
 * What the list promises depends on [kind]. A [RouteKind.Direct] route names what runs: every unit of work
 * reaches every component. A [RouteKind.Composed] route names what may run, because the call's windows are
 * not all the same length and so do not all reach the same implementation, and [reason] says so.
 * [RouteKind.NoWork] is a call whose own contract stops before the arithmetic, where nothing but the
 * destination scaling runs.
 *
 * [executionGroup] is how many logical columns the backend recommended handing over at a time for this call,
 * and zero where the call schedules no panel. It is a grouping, not a lane count, and the two are different
 * numbers that agree only by coincidence.
 *
 * Building a route inspects the shape, so it belongs before a timed region and never inside one.
 */
public class DenseMatrixRoute internal constructor(
    /** The operation requested. */
    public val operation: DenseMatrixOperation,
    /** How the call was served. */
    public val kind: RouteKind,
    /** The component that owns traversal, windows and arithmetic order. */
    public val scheduling: String,
    /** The resolved entry point within [scheduling]. */
    public val entryPoint: String,
    /** Every panel implementation and Level 1 leaf the call reaches, in call order. */
    public val components: List<String>,
    /** Logical columns the backend recommended per group, or zero where the call schedules no panel. */
    public val executionGroup: Int,
    /** What the components cover, what the traversal keeps for itself, or why the call is composed. */
    public val reason: String?,
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

    override fun toString(): String = buildString {
        append(operation.name.lowercase())
        append(' ')
        append(kind.name.lowercase())
        append(' ')
        append(implementation)
        append(' ')
        append(entryPoint)
        if (executionGroup > 0) append(" group=").append(executionGroup)
        reason?.let { append(" (").append(it).append(')') }
    }
}
