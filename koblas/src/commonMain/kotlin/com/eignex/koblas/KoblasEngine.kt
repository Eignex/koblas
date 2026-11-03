@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.dense.*
import com.eignex.koblas.internal.portable.sparse.*

/**
 * Dense and sparse BLAS through one immutable engine.
 *
 * Each BLAS routine is one member. The portable engine runs the shared schedule over its own kernels; a
 * composed engine such as [koblas] routes each call to the provider its profile selects and otherwise runs the
 * same schedule over its own kernels; a vendor engine from [BuiltinEngines.vendor] calls its library for the
 * routines it exports and fails for the rest. [BuiltinEngines] names exact engines, while [koblas] selects its
 * standard platform configuration once.
 *
 * Level 2 and 3 routines validate shapes, preserve BLAS no-read rules and snapshot an input that shares the
 * destination's storage, borrowing from a [Scratch] when one is given. Level 1 members address primitive array
 * windows that must be valid; callers stage destructive overlaps. Rounding, reduction order and unspecified
 * exceptional results belong to the selected arithmetic.
 */
public interface KoblasEngine {

    /** Implementation identifier for diagnostics. */
    public val name: String

    /** Stable identifier used by routing profiles. */
    public val id: String get() = vendor?.vendorName?.lowercase() ?: name

    /** Callbacks supplied by this provider, trusted by the composer. Shared matrix schedules select arithmetic directly from leaf [providers]. */
    public val implementations: Set<EngineImplementation> get() = emptySet()

    /** Library provenance, when applicable. */
    public val vendor: Vendor? get() = null

    /** Resolved host library, when applicable. */
    public val libraryPath: String get() = ""

    /** Host library version, when applicable. */
    public val version: String get() = ""

    /** Evidence for the host library's compute thread count. */
    public val threadEvidence: ThreadEvidence get() = ThreadEvidence.Unconfirmed

    /** Returns `sum(a[aOff + i] * b[bOff + i])` over [len] entries, or zero for an empty run. */
    public fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Double = unsupported()

    /**
     * Adds `alpha * x[xOff + i]` into `y[yOff + i]` over [len] entries. A zero [alpha] returns without
     * evaluating the products, as required by standalone BLAS AXPY semantics.
     */
    public fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int = 1,
        xStride: Int = 1,
    ): Unit = unsupported()

    /** Scales [len] entries in [v] by [alpha]. */
    public fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int = 1): Unit = unsupported()

    /**
     * Returns the rescaled Euclidean norm over [len] entries, or zero for an empty run. Implementations
     * must avoid the avoidable overflow and underflow of a plain square-sum followed by a square root.
     */
    public fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    public fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    /**
     * Zero-based index within the selected run of a maximum absolute value, or `-1` when empty. A nonempty
     * run of zeros returns `0`.
     *
     * Which index is returned when the maximum is not unique, and what happens when the largest magnitude is
     * a NaN, is this implementation's. `idamax` specifies neither: the portable kernels compare strictly, so
     * ties go to the first and a NaN loses to a later finite entry, while a vendor may report the NaN's index
     * and may resolve a tie the other way for a negatively spaced run.
     */
    public fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Int = unsupported()

    /**
     * Applies the plane rotation `(c, s, -s, c)` to two primitive vector runs. Each pair is loaded before either
     * result is stored. Equal runs are therefore safe, but callers must snapshot any other overlap whose
     * stores could precede a later input load.
     */
    @Suppress("LongParameterList")
    public fun rot(
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        len: Int,
        c: Double,
        s: Double,
        xStride: Int = 1,
        yStride: Int = 1,
    ): Unit = unsupported()

    /** Exchanges [len] entries of the two runs. */
    public fun swap(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = unsupported()

    /** Copies a primitive vector window. Callers stage destructive overlap before dispatch. */
    public fun copy(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = unsupported()

    /**
     * Returns the plain sum over [len] entries, or zero for an empty run. This is not compensated summation;
     * compiled leaves may reduce vector lanes as a tree.
     */
    public fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    /** Dot product of the stored entries and their addressed dense positions. The caller validates and stages aliases before dispatch. */
    public fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = unsupported()

    /** Dot product of matching stored positions in two sorted sparse runs. The caller validates and stages aliases before dispatch. */
    public fun dotSparse(
        xIndices: IntArray,
        xIndexOffset: Int,
        xValues: DoubleArray,
        xValueOffset: Int,
        xCount: Int,
        yIndices: IntArray,
        yIndexOffset: Int,
        yValues: DoubleArray,
        yValueOffset: Int,
        yCount: Int,
    ): Double = unsupported()

    /** Adds the scaled stored run at its addressed dense positions. */
    public fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ): Unit = unsupported()

    /** Copies stored values to their addressed dense positions. The caller validates and stages aliases before dispatch. */
    public fun computeIndexedScatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = unsupported()

    /** Copies addressed dense entries to the stored value run. The caller validates and stages aliases before dispatch. */
    public fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = unsupported()

    /** Gathers addressed entries and replaces those dense positions with zero. The caller validates and stages aliases before dispatch. */
    public fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = unsupported()

    /** Euclidean norm over the addressed dense positions. The caller validates and stages aliases before dispatch. */
    public fun computeIndexedNrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        unsupported()

    /**
     * `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose]. `x` is the
     * [xStride]-spaced window of [x] from [xOffset], and `y` likewise; a negative stride walks backwards from its
     * offset, which is the window's first entry.
     *
     * A zero [alpha] or an empty `A` reads neither `A` nor `x`, and a zero [beta] overwrites `y` without reading
     * it. An operand sharing `y`'s storage is snapshotted, from [scratch] when one is given, before the first
     * write.
     */
    public fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        transpose: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledGemv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch?.arrays)

    /**
     * `y = alpha · A · x + beta · y` for a symmetric `A` (BLAS `dsymv`), read from its [lower] or upper triangle,
     * diagonal included. Windows, zero multipliers and overlap follow [gemv].
     */
    public fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ): Unit = scheduledSymv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch?.arrays)

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`). A vector sharing `A`'s storage is snapshotted. */
    public fun ger(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        scratch: Scratch? = null,
    ): Unit = scheduledGer(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch?.arrays)

    /** `A = A + alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. Overlap follows [ger]. */
    public fun syr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ): Unit = scheduledSyr(alpha, x, xOffset, xStride, a, lower, scratch?.arrays)

    /** `A = A + alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. */
    public fun syr2(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ): Unit = scheduledSyr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, lower, scratch?.arrays)

    /**
     * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a], `op`
     * transposing when [transpose] and [unitDiag] taking the diagonal as 1. The window of [x] carries `b` in and
     * `x` out.
     *
     * As in BLAS `dtrsv`, no singularity or conditioning check is performed; singular inputs have no guaranteed
     * numerical result. The substitution overwrites `x` as it goes, so a triangle sharing that storage is
     * snapshotted first.
     */
    public fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledTrsv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch?.arrays)

    /** `x = op(T) · x` in place (BLAS `dtrmv`), with the flags and overlap of [trsv]. */
    public fun trmv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledTrmv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch?.arrays)

    /**
     * `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *
     * An input that shares [c]'s storage is snapshotted first. A zero [alpha] or an empty shared dimension
     * scales [c] and reads no operand; a zero [beta] overwrites [c] without reading it.
     *
     * Scaling placement and accumulation order depend on the selected implementation, including whether
     * [alpha] scales operand entries or partial sums, so rounding, overflow and non-finite results may differ
     * between implementations and transpose modes.
     */
    public fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        scratch: Scratch? = null,
    ): Unit = scheduledGemm(alpha, a, transposeA, b, transposeB, beta, c, scratch?.arrays)

    /**
     * `C = alpha · op(A) · op(B) + beta · C` in only the selected triangle (Netlib `GEMMTR`, commonly exposed as
     * `gemmt`). `op(A)` is `n×k`, `op(B)` is `k×n`, and [c] is `n×n`. The opposite triangle is neither read nor
     * written. Overlap and multipliers follow [gemm].
     */
    public fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ): Unit = scheduledGemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, scratch?.arrays)

    /**
     * `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
     * [lower] or upper triangle of the symmetric [a] is read. Overlap and multipliers follow [gemm].
     */
    public fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        right: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledSymm(alpha, a, b, beta, c, lower, right, scratch?.arrays)

    /**
     * `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`). Only the
     * [lower] or upper triangle is written. One product, so [alpha] and the accumulation order carry the
     * latitude [gemm] describes.
     */
    public fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ): Unit = scheduledSyrk(alpha, a, transpose, beta, c, lower, scratch?.arrays)

    /**
     * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     * [transpose]. Only the [lower] or upper triangle is written. The implementation may accumulate the two
     * products separately or in one traversal, with the rounding differences [gemm] describes.
     */
    public fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ): Unit = scheduledSyr2k(alpha, a, b, transpose, beta, c, lower, scratch?.arrays)

    /**
     * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`). Flags follow
     * [trsv]; the right-hand sides are the columns of [b] from the left and its rows from the right.
     */
    public fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ): Unit = scheduledTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true, scratch?.arrays)

    /** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of [trsm]. */
    public fun trmm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ): Unit = scheduledTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false, scratch?.arrays)

    /**
     * In-place `y = alpha · op(A) · x + beta · y` for a sparse [a], where `op(A)` is `Aᵀ` when [transpose]. A zero
     * [beta] overwrites [y] without reading it, and a zero [alpha] reads neither [a] nor [x].
     *
     * Only stored entries are visited, so an implicit zero of [a] forms no product: a column [a] does not store
     * leaves [y] untouched even where [x] holds an infinity. A stored zero does form its product. [x] and [a]'s
     * coefficients may share [y]'s storage and are snapshotted before [y] is scaled.
     */
    public fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseGemv(alpha, a, x, beta, y, transpose, scratch?.arrays)

    /**
     * `y = alpha · A · x + beta · y` for a square sparse [a] interpreted as symmetric. Exactly the selected
     * [lower] or upper stored triangle is read; a stored off-diagonal entry contributes to both mirrored
     * positions. Multipliers and overlap follow the sparse [gemv].
     */
    public fun symv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseSymv(alpha, a, x, beta, y, lower, scratch?.arrays)

    /**
     * `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right], interpreting the square
     * sparse [a] as symmetric from exactly its selected [lower] or upper triangle. A zero [alpha] reads neither
     * input and a zero [beta] does not read [c]. Shared input and destination storage is staged.
     */
    public fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        right: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseSymm(alpha, a, b, beta, c, lower, right, scratch?.arrays)

    /**
     * Solve `op(T) · x = b` in place for a sparse triangle, with the flags of the dense [trsv]. A missing diagonal
     * entry represents zero and no singularity check is performed. A right-hand side that is already exactly zero
     * contributes no update, as in reference `dtrsv`.
     */
    public fun trsv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseTrsv(a, x, lower, transpose, unitDiag, scratch?.arrays)

    /** `x = op(T) · x` in place for a sparse triangle. A missing diagonal entry is simply zero. */
    public fun trmv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseTrmv(a, x, lower, transpose, unitDiag, scratch?.arrays)

    /**
     * `C = alpha · op(A) · op(B) + beta · C` (Sparse BLAS `usmm`) for a sparse [a] against a dense [b], or
     * `C = alpha · op(B) · op(A) + beta · C` when [right]. A zero [beta] overwrites [c] without reading it.
     *
     * [right] exists because [a] is distinguished by being the sparse operand and cannot be swapped, so without a
     * side the product of a dense matrix by a sparse one would only be reachable by transposing the result.
     */
    public fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        right: Boolean = false,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseDenseGemm(alpha, a, transposeA, b, transposeB, beta, c, right, scratch?.arrays)

    /**
     * Fresh CSC `alpha · op(A) · op(B)`. Product support is discovered from stored positions, including explicit
     * zeros, retained cancellations and underflowed values; rows are sorted within every column and all result
     * arrays are owned. A zero [alpha] discovers the same structure without reading operand values.
     */
    public fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
    ): SparseMatrix = scheduledSparseProduct(alpha, a, transposeA, b, transposeB)

    /**
     * Dense-destination `C = alpha · op(A) · op(B) + beta · C` for two sparse operands. No sparse intermediate is
     * created and implicit zeros are not evaluated. A zero [alpha] reads neither operand's values and a zero
     * [beta] does not read [c].
     */
    public fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseProductInto(alpha, a, transposeA, b, transposeB, beta, c, scratch?.arrays)

    /**
     * Dense selected-triangle `C = alpha · op(A) · op(A)ᵀ + beta · C` for a sparse [a]. Only the [lower] or upper
     * triangle of [c] is read or written, and products involving implicit zeros are not evaluated.
     */
    public fun syrk(
        alpha: Double,
        a: SparseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseSyrk(alpha, a, transpose, beta, c, lower, scratch?.arrays)

    /**
     * Fresh CSC selected triangle of `op(A) · op(A)ᵀ`. Stored-product support and cancellations are retained, rows
     * are sorted, and result arrays are owned. The opposite triangle is absent rather than implicitly mirrored.
     */
    public fun syrk(a: SparseMatrix, transpose: Boolean = false, lower: Boolean = true): SparseMatrix =
        symmetricRankProduct(a, transpose, lower)

    /**
     * Fresh CSC `alpha · op(A) + B`. The structural union is retained, including explicit zeros, cancellations
     * and positions only `A` stores when [alpha] is zero, in which case `A`'s values are not read.
     */
    public fun addScaled(alpha: Double, a: SparseMatrix, transposeA: Boolean, b: SparseMatrix): SparseMatrix =
        scheduledSparseAddScaled(alpha, a, transposeA, b)

    /**
     * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (Sparse BLAS `ussm`), with the
     * flags of the sparse [trsv]. A missing diagonal entry behaves as zero and singularity is not reported. From
     * the right a stored zero off the diagonal is skipped, as reference `dtrsm` skips a zero coefficient.
     */
    public fun trsm(
        a: SparseMatrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseTrsm(a, b, lower, transpose, unitDiag, right, alpha, scratch?.arrays)

    /**
     * `B = alpha · op(T) · B` in place, or `B = alpha · B · op(T)` when [right]. Zero or missing diagonal entries
     * participate normally. From the right a stored zero off the diagonal is skipped, as reference `dtrmm` skips
     * a zero coefficient.
     */
    public fun trmm(
        a: SparseMatrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ): Unit = scheduledSparseTrmm(a, b, lower, transpose, unitDiag, right, alpha, scratch?.arrays)
}
