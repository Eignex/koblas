package com.eignex.koblas.dense

import com.eignex.koblas.MatrixRoute
import com.eignex.koblas.RouteKind
import com.eignex.koblas.vendor.CallRoute

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

    /** Rank-one update of a rectangle within an existing matrix, using the owned panel kernels. */
    GerWindow("ger-window"),

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
 * Facts used by expert shape diagnostics, inspected without executing arithmetic.
 *
 * Rows and columns describe stored operands for Level 2, destination extents for Level 3 and the selected
 * rectangle for windowed GER. Level 3 requires depth. Transpose/side/triangle flags match the call.
 * Contiguous describes a shared Level 2 vector; leading dimensions describe borrowed matrix storage.
 * Aliased and borrowed expose staging costs. Product operand strides follow their stored orientation.
 *
 * @property rows Stored rows or destination rows for Level 3.
 * @property columns Stored columns or destination columns for Level 3.
 * @property alpha Operand multiplier.
 * @property beta Destination multiplier.
 * @property contiguous Whether a shared Level 2 vector has unit stride.
 * @property depth Shared product dimension or structured order; required for Level 3.
 * @property lower Selected stored triangle.
 * @property transposeA Whether the left operand is transposed.
 * @property transposeB Whether the right operand is transposed.
 * @property right Whether the structured operand is on the right.
 * @property aliased Whether an input overlaps destination storage.
 * @property borrowed Whether matrix operands include borrowed regions.
 * @property leadingDimension Destination column spacing, or null for contiguous storage.
 * @property rightLeadingDimension Stored column spacing of a borrowed right operand.
 * @property leftLeadingDimension Stored column spacing of a borrowed left operand.
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
    public val borrowed: Boolean = false,
    public val leadingDimension: Int? = null,
    public val rightLeadingDimension: Int? = null,
    public val leftLeadingDimension: Int? = null,
) {
    init {
        require(rows >= 0) { "negative row count" }
        require(columns >= 0) { "negative column count" }
        require(leftLeadingDimension == null || leftLeadingDimension >= 0) { "negative left leading dimension" }
        require(rightLeadingDimension == null || rightLeadingDimension >= 0) { "negative right leading dimension" }
        require(leadingDimension == null || leadingDimension >= rows) { "leading dimension below row count" }
        require(depth == null || depth >= 0) { "negative product depth" }
    }
}

/** Shape diagnostics. Components name reached bodies; composed routes may also execute scalar traversal. */
public class DenseMatrixRoute internal constructor(
    /** The operation requested. */
    public val operation: DenseMatrixOperation,
    /** How the call was served. */
    override val kind: RouteKind,
    /** The component that owns traversal and windows. */
    override val scheduling: String,
    /** The resolved entry point within [scheduling]. */
    override val entryPoint: String,
    /** Every panel implementation and Level 1 leaf the call reaches, in call order. */
    override val components: List<String>,
    /** Logical columns the panel backend recommended per group, or zero where no panel runs. */
    override val executionGroup: Int,
    /**
     * The whole-call vendor route this call was handed to, or null when no single entry point served it.
     *
     * Null is not the same as no library: a portable schedule may still call one for a window of work, and
     * [components] names that leaf where it does.
     */
    public val host: CallRoute? = null,
) : MatrixRoute {
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
    }
}
