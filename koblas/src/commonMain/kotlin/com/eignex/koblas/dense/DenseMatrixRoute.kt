package com.eignex.koblas.dense

import com.eignex.koblas.vendor.CallRoute
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

    /** The same product between two operands that were packed for the tile before the call. */
    GemmPacked("gemm-packed"),

    /** The same product with the left operand retained and the right packed for this call. */
    GemmPackedLeft("gemm-packed-left"),

    /** The same product with the right operand retained and the left packed for this call. */
    GemmPackedRight("gemm-packed-right"),

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
 *   strided operand is scalar work at any width, so it is part of the question. A product does not read it:
 *   both of its operands are whole matrices, so what its panels share follows from the extents and the
 *   transpose flags and is worked out rather than declared.
 * @property depth the inner extent of a product, or null when the operation has none. Zero is a product over
 *   nothing, which is a call with nothing to do however large its operands are, so it is nullable rather than
 *   zero-defaulted. A product operation requires it: which route such a call takes is a question about all
 *   three extents, and a missing one is refused rather than read as none.
 * @property lower which triangle a selected-triangle operation reads, ignored by the rest. Part of the
 *   question because the windows a triangular traversal cuts are not the same on the two sides once a group
 *   covers more than one column.
 * @property transposeA whether a product's left operand is transposed, ignored outside one. A product small
 *   enough to run without packing is a reduction down stored columns when it is and an accumulation of them
 *   when it is not, which are different panels.
 * @property transposeB whether a product's right operand is transposed, ignored outside one. It decides
 *   whether the coefficient vector an unpacked product shares across a panel is adjacent or strided, and a
 *   strided one is scalar work at any width.
 * @property right whether a two-sided operation has its structured operand on the right, ignored by the rest.
 *   It decides which extent of the destination the right-hand sides are counted along and so how far apart
 *   they lie, which is what a substitution over them has to work with. A unit diagonal is deliberately not
 *   here: it removes a division from every step and changes no body any window reaches, so a route that took
 *   it would be distinguishing calls that execute the same way.
 * @property aliased whether an input operand shares the destination's backing buffer. A built-in schedule
 *   stages a copy of it and reaches the same bodies either way, so the portable route ignores this; a
 *   composition that hands the whole call to a host library cannot, since a whole-call binding takes the
 *   caller's storage as it lies and has no argument for saying that two of its operands are one buffer. The
 *   copy is a component of that call, and a route that omitted it would be describing a different call.
 */
@Suppress("LongParameterList") // a call is the facts that decide what runs, each of which changes the answer
public class DenseCall(
    public val rows: Int,
    public val columns: Int,
    public val alpha: Double = 1.0,
    public val beta: Double = 1.0,
    public val contiguous: Boolean = true,
    public val depth: Int? = null,
    public val lower: Boolean = true,
    public val transposeA: Boolean = false,
    public val transposeB: Boolean = false,
    public val right: Boolean = false,
    public val aliased: Boolean = false,
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
 * [executionGroup] is how many logical columns the backend recommended handing over at a time for the Level
 * 2 panel work this call schedules, and zero where it schedules none or where the windows it cuts did not
 * agree on one. It is a grouping, not a lane count, and the two are different numbers that agree only by
 * coincidence. A triangular matrix call groups its right-hand sides as well, which is a separate number
 * chosen by a separate backend; that one is named in [reason] rather than here, because a field holding
 * whichever of the two was asked for last would describe neither.
 *
 * [host] is the whole-call vendor route underneath a call an engine handed over entire, and it is a narrower
 * statement than it looks. Its absence says that no single vendor entry point served this call; it does not
 * say that no library ran. Kotlin/Native's default gives its portable dense schedule the vendor's Level 1
 * kernels, so a call that stayed on that schedule can still reach a library for a window of work, and
 * [components] is where such a leaf is named. Reading the two together is what distinguishes a call handed
 * over whole from one this library scheduled and handed pieces of.
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
    /** Logical columns the panel backend recommended per group, or zero where no panel runs. */
    public val executionGroup: Int,
    /** What the components cover, what the traversal keeps for itself, or why the call is composed. */
    public val reason: String?,
    /**
     * The whole-call vendor route this call was handed to, or null when no single entry point served it.
     *
     * Null is not the same as no library: a portable schedule may still call one for a window of work, and
     * [components] names that leaf where it does.
     */
    public val host: CallRoute? = null,
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
        host?.let { append(" host=").append(it) }
        reason?.let { append(" (").append(it).append(')') }
    }
}
