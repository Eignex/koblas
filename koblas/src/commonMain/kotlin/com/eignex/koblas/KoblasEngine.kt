@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.configuration.*
import com.eignex.koblas.internal.dense.*
import com.eignex.koblas.vendor.*

/**
 * One immutable contract for portable, SIMD, vendor and combined dense and sparse BLAS.
 *
 * Providers implement only the parts they supply and declare [implementations]. Unsupported methods
 * fail without calling another provider; [combine] explicitly composes complete coverage. [BuiltinEngines] selects exact
 * engines, while [koblas] selects its standard platform configuration once.
 *
 * Matrix and container operations validate shapes, preserve BLAS no-read rules and stage supported aliases.
 * Primitive array windows require valid offsets, lengths and strides; callers stage destructive overlaps.
 * Rounding, reduction order and unspecified exceptional results belong to the selected arithmetic.
 */
public interface KoblasEngine {

    /** Implementation identifier for diagnostics. */
    public val name: String

    /** Stable identifier used by routing profiles. */
    public val id: String get() = vendor?.vendorName?.lowercase() ?: name

    /** Arithmetic supplied by this provider, trusted by the composer. */
    public val implementations: Set<EngineImplementation> get() = emptySet()

    /** Library provenance, when applicable. */
    public val vendor: Vendor? get() = null

    /** Resolved host library, when applicable. */
    public val libraryPath: String get() = ""

    /** Host library version, when applicable. */
    public val version: String get() = ""

    /** Evidence for the host library's compute thread count. */
    public val threadEvidence: ThreadEvidence get() = ThreadEvidence.Unconfirmed

    /** Whether attribution can depend on the values processed by this contribution. */
    public fun dynamicAttribution(implementation: EngineImplementation): Boolean = false

    /** Whether this contribution requires unit strides. */
    public fun requiresUnitStride(implementation: EngineImplementation): Boolean = false

    /** Smallest run the contribution can execute. The composer checks it before calling the provider. */
    public fun minimumLength(implementation: EngineImplementation): Int = 0

    /** Returns `sum(a[aOff + i] * b[bOff + i])` over [len] entries, or zero for an empty run. */
    public fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Double = unsupportedEngineOperation(name, "dot")

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
    ): Unit = unsupportedEngineOperation(name, "axpy")

    /** Scales [len] entries in [v] by [alpha]. */
    public fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int = 1): Unit =
        unsupportedEngineOperation(
            name,
            "scale",
        )

    /**
     * Returns the rescaled Euclidean norm over [len] entries, or zero for an empty run. Implementations
     * must avoid the avoidable overflow and underflow of a plain square-sum followed by a square root.
     */
    public fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupportedEngineOperation(
        name,
        "nrm2",
    )

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    public fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupportedEngineOperation(
        name,
        "asum",
    )

    /**
     * Zero-based index within the selected run of a maximum absolute value, or `-1` when empty. A nonempty
     * run of zeros returns `0`.
     *
     * Which index is returned when the maximum is not unique, and what happens when the largest magnitude is
     * a NaN, is this implementation's. `idamax` specifies neither: the portable kernels compare strictly, so
     * ties go to the first and a NaN loses to a later finite entry, while a vendor may report the NaN's index
     * and may resolve a tie the other way for a negatively spaced run.
     */
    public fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Int = unsupportedEngineOperation(
        name,
        "iamax",
    )

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
    ): Unit = unsupportedEngineOperation(name, "rot")

    /** Exchanges [len] entries of the two runs. */
    public fun swap(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = unsupportedEngineOperation(name, "swap")

    /** Copies a primitive vector window. Callers stage destructive overlap before dispatch. */
    public fun copy(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = unsupportedEngineOperation(name, "copy")

    /**
     * Returns the plain sum over [len] entries, or zero for an empty run. This is not compensated summation;
     * compiled leaves may reduce vector lanes as a tree.
     */
    public fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupportedEngineOperation(
        name,
        "sum",
    )

    /** Dot product of the stored entries and their addressed dense positions. The caller validates and stages aliases before dispatch. */
    public fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = unsupportedEngineOperation(name, "dotDense")

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
    ): Double = unsupportedEngineOperation(name, "dotSparse")

    /** Adds the scaled stored run at its addressed dense positions. */
    public fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ): Unit = unsupportedEngineOperation(name, "axpy")

    /** Copies stored values to their addressed dense positions. The caller validates and stages aliases before dispatch. */
    public fun computeIndexedScatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = unsupportedEngineOperation(name, "computeIndexedScatter")

    /** Copies addressed dense entries to the stored value run. The caller validates and stages aliases before dispatch. */
    public fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = unsupportedEngineOperation(name, "gather")

    /** Gathers addressed entries and replaces those dense positions with zero. The caller validates and stages aliases before dispatch. */
    public fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = unsupportedEngineOperation(name, "gatherZero")

    /** Euclidean norm over the addressed dense positions. The caller validates and stages aliases before dispatch. */
    public fun computeIndexedNrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        unsupportedEngineOperation(
            name,
            "computeIndexedNrm2",
        )

    /**
     * How many columns of [work] this engine recommends processing together, for a panel of [rows] by
     * [columns].
     *
     * Always at least one and never more than [columns] when [columns] is positive. The answer is a local
     * tuning choice made from the resolved vector width and small measured rules, and it is not a lane count:
     * a group of four columns and four lanes are different numbers that happen to agree on some machines.
     *
     * [contiguous] is part of the question for the same reason it is part of [implementationFor]'s: an engine
     * whose body over adjacent entries is a different body may want a different amount of work in it. A
     * caller that will hand over a strided window has to say so, or it is given a recommendation for a body
     * it is not going to reach. A scheduling of its own may then take less than what was recommended, for a
     * cache or a workspace it is keeping within; passing less is always correct.
     */
    public fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean = true): Int =
        unsupportedEngineOperation(
            name,
            "executionGroup",
        )

    /**
     * Whether a panel of these extents is worth copying into adjacent rows before it is handed over.
     *
     * A caller that can copy a strided window into adjacent storage needs to know whether the copy buys
     * anything before it pays for it, and the answer is the engine's, because only the engine knows what
     * its body does with the two layouts. A portable body reads one entry at a time either way and says no
     * at every extent. A vector body says yes where the panel is wide enough that the arithmetic it gains
     * outweighs a pass over the data, which is not the same width as the one where its body starts running:
     * one lane block is a single vector operation per column, and a copy is not repaid by that.
     *
     * This is a recommendation about copying rather than a claim about which body runs, and the two are
     * asked separately. Where it is true, [implementationFor] also names a different body for the two
     * layouts; where it is false the body may still differ, and the caller has simply been told not to pay
     * for the difference.
     */
    public fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean = unsupportedEngineOperation(
        name,
        "prefersContiguous",
    )

    /**
     * `y(c) = alpha · Σ a(i, c) · x(i) + beta · y(c)` for each of [columns] columns.
     *
     * The destination is [columns] entries from [yOffset], spaced [yStride] apart. A zero [beta] overwrites
     * without reading what is there, as every BLAS routine carrying one does. With no rows every sum is
     * empty and so zero, and each output is still written from that and whatever [beta] contributes; with no
     * columns nothing happens at all.
     */
    public fun multiDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ): Unit = unsupportedEngineOperation(name, "multiDot")

    /**
     * `y(i) += Σ (alpha · x(c)) · a(i, c)` over [rows] destination entries and [columns] panel columns.
     *
     * Scaling and reduction grouping may vary between implementations; grouping does not promise identical
     * rounding or exceptional results.
     */
    public fun columnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ): Unit = unsupportedEngineOperation(name, "columnUpdate")

    /**
     * One pass over a panel that both updates a destination window and reduces the panel against a vector.
     *
     * `y(i) += Σ (alpha · coefficients(c)) · a(i, c)` over the [rows] entries of the window, and
     * `sums(c) += alpha · Σ a(i, c) · x(i)` for each column. This is a symmetric traversal's second half
     * removed: the stored triangle is read once and serves both the column below the diagonal and the row
     * to its left.
     *
     * [sums] is the caller's, which is where several results go rather than into a returned object. Its
     * window must not overlap [y]'s, which for a symmetric traversal it never does: the reduction lands on
     * the diagonal side of the rows the update wrote.
     */
    public fun coupledUpdateDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
        xStride: Int = 1,
        yStride: Int = 1,
        coefficientStride: Int = 1,
        sumStride: Int = 1,
    ): Unit = unsupportedEngineOperation(name, "coupledUpdateDot")

    /**
     * `a(i, c) += (alpha · coefficients(c)) · x(i)`, the rank-one update of a panel of destination columns.
     *
     * The destination is the panel here and [x] is the source window, which is the opposite direction from
     * [columnUpdate] and the reason it is its own entry rather than that one with the arguments swapped.
     */
    public fun rankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        coefficientStride: Int,
    ): Unit = unsupportedEngineOperation(name, "rankUpdate")

    /**
     * `a(i, c) += alpha · (x(i) · u(c) + y(i) · v(c))`, the symmetric rank-two update of a panel.
     *
     * Two [rankUpdate] calls compute the same thing, and that is what this does unless an engine overrides it.
     * One pass loads and stores each destination entry once instead of twice, which is the limit a narrow
     * vector unit reaches first.
     */
    public fun rankTwoUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        rows: Int,
        columns: Int,
        u: DoubleArray,
        uOffset: Int,
        uStride: Int,
        v: DoubleArray,
        vOffset: Int,
        vStride: Int,
    ): Unit = unsupportedEngineOperation(name, "rankTwoUpdate")

    /**
     * `y(i) += Σ (alpha · values(c)) · a(i, indices(c))` over a panel whose columns an index array selects.
     *
     * The sparse counterpart of [columnUpdate], and the shape a sparse column takes when its dense operand
     * carries several right-hand sides. [rows] is the group of right-hand sides, which is the axis with no
     * dependences between its entries and therefore the one an engine may vectorise; [columns] is how many
     * stored entries the column contributes, read as `indices` and `values` from [fromIndex] onward.
     *
     * An entry of the panel is at `aOffset + i · rowStride + indices(c) · indexStride`, which is two strides
     * rather than a leading dimension because either axis may be the contiguous one. A dense block in
     * column-major storage has its right-hand sides a leading dimension apart, and one staged into
     * right-hand-side-major order has them adjacent; the arithmetic is the same and only the strides differ.
     * [implementationFor] answers which body a given panel reaches, and a caller that wants the vector one
     * has to make [rowStride] one for it.
     *
     * The destination is [rows] adjacent entries from [yOffset] and is accumulated into. Each indexed stored
     * position participates, including a stored zero; absent positions are never evaluated. This sparse
     * structural rule also applies to [indexedRankUpdate] and [indexedCoupledUpdate], independently of the
     * zero-coefficient latitude of dense updates.
     */
    public fun indexedColumnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
    ): Unit = unsupportedEngineOperation(name, "indexedColumnUpdate")

    /**
     * `a(i, indices(c)) += (alpha · values(c)) · x(i)`, the indexed counterpart of [rankUpdate].
     *
     * The destination is the panel and [x] is the source window of [rows] adjacent entries, addressed as
     * [indexedColumnUpdate] describes.
     *
     * Every logical entry this call writes must have an address of its own. Two things are needed for that
     * and the caller owns both: the selected positions must be distinct, which a validated CSC column's
     * ascending row indices are, and the two strides must describe a panel that does not fold onto itself,
     * which they fail to do where a stride is zero or where one is a multiple of the other inside the
     * extents in play. An engine may hold several entries in registers at once, so two logical entries
     * sharing an address would lose an update rather than apply both. This is the one precondition these two
     * leaves do not share: a reduction may read a position twice, a scatter may not write one twice.
     *
     * These internal primitive windows are trusted; validation belongs to the common matrix entry points. A public matrix call passes windows of its
     * own storage, where the strides come from a layout that already holds.
     */
    public fun indexedRankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        x: DoubleArray,
        xOffset: Int,
    ): Unit = unsupportedEngineOperation(name, "indexedRankUpdate")

    /**
     * One pass over an indexed panel that both scatters into the positions it selects and reduces them.
     *
     * For every position `c` of the run, with `t = alpha · values(c)`:
     * `a(i, indices(c)) += t · b[pivot + i · rowStride]`, and `sums(i) += t · b(i, indices(c))` unless `c`
     * is [excluded].
     *
     * This is [indexedRankUpdate] and [indexedColumnUpdate] over one walk of the same run, and the indexed
     * counterpart of [coupledUpdateDot]. It is here for the same reason that one is: a symmetric traversal
     * stores one triangle, every stored entry serves the column it sits in and the row it mirrors into, and
     * reading it once is half the index work of reading it twice. One multiplier serves both halves, so a
     * caller applies its own scaling through [alpha] rather than pre-scaling one side of the pair.
     *
     * Both halves read [b]: the scattered one reads the [rows] entries from [pivot] spaced [rowStride]
     * apart, which is one indexed column of the same window, and the reduced one reads the column each
     * position selects. A caller therefore hands over no coefficients of its own, because gathering that
     * column into a scratch first would be a pass over it per column of the sparse operand.
     *
     * [a] and [b] are windows of the same shape addressed alike, from [offset] with the two strides
     * [indexedColumnUpdate] describes; a symmetric product passes its destination and its source, which are
     * different arrays. The reduction lands in [rows] entries from [sumOffset] spaced [rowStride] apart.
     * This window may coincide with one indexed column of [a], with each reduction entry overlapping only
     * the scatter entry for the same logical right-hand side; an overlap between different right-hand sides
     * is unsupported. Where both updates target the same entry, the scatter is applied before the reduction
     * for each stored position. All source windows remain disjoint from both destination windows.
     *
     * [excluded] is a position relative to this run, in `0 until columns`, whose product is scattered but
     * not reduced; `-1` excludes none. The column of [b] it selects is not read, though [pivot] still is,
     * because the scatter runs at that position like any other. A symmetric column excludes its own diagonal
     * because that contribution is already scattered into the pivot row.
     *
     * Stored positions participate even when their values are zero. The reduction accumulates into its
     * destination; term grouping and rounding may vary subject to the overlap rule above.
     */
    @Suppress("LongParameterList") // the panel, its two windows, the run, and both results
    public fun indexedCoupledUpdate(
        alpha: Double,
        a: DoubleArray,
        b: DoubleArray,
        offset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        pivot: Int,
        sums: DoubleArray,
        sumOffset: Int,
        excluded: Int,
        destinationOffset: Int = offset,
        destinationRowStride: Int = rowStride,
        destinationIndexStride: Int = indexStride,
    ): Unit = unsupportedEngineOperation(name, "indexedCoupledUpdate")

    /**
     * Destination rows one register tile accumulates, which is the group a left panel is packed in.
     *
     * Positive and fixed for the life of an implementation.
     */
    public val tileRows: Int get() = unsupportedEngineOperation(name, "tileRows")

    /** Destination columns one register tile accumulates, which is the group a right panel is packed in. */
    public val tileColumns: Int get() = unsupportedEngineOperation(name, "tileColumns")

    /**
     * Whether a product of these logical extents is worth packing into this engine's tiles at all.
     *
     * Packing costs a pass over both operands and buys contiguous, tile-shaped reads for the whole product.
     * The arithmetic grows with `rows · columns · depth` and the copy with `rows · depth + depth · columns`,
     * so a product with any small extent pays for a copy it cannot amortise, and a matrix-vector shaped call
     * pays for one it can never amortise at all. Where this answers false the caller runs the product as
     * panel work over the operands where they are, which is what avoids copying both of them into tiles. It
     * is not a promise that nothing is copied: that path may still gather one strided coefficient column,
     * borrow a destination column to accumulate into, and stage an operand that shares the destination.
     *
     * The crossover is local: a tile that finishes more arithmetic per loaded value pays off a copy sooner.
     */
    public fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean = unsupportedEngineOperation(
        name,
        "packsProduct",
    )

    /**
     * `C = alpha · A · B + beta · C` over one `rows` by `columns` window, accumulating `depth` steps.
     *
     * [packedA] holds the left operand in [tileRows]-wide groups and [packedB] the right in
     * [tileColumns]-wide ones. Within a group, depth steps are adjacent blocks of that width, so entry
     * `(i, p)` of the left operand sits at `aOffset + (i / tileRows) · aGroupStride + p · tileRows +
     * i % tileRows`, and entry `(p, j)` of the right at `bOffset + (j / tileColumns) · bGroupStride +
     * p · tileColumns + j % tileColumns`.
     *
     * The group strides are `depth · tileRows` and `depth · tileColumns` for transient panels.
     *
     * The destination window starts at [cOffset] with columns [ldc] apart. With no rows or no columns
     * nothing happens and nothing is read. With no depth the product is empty, so the window becomes
     * `beta · C` and neither packed panel is read.
     */
    public fun productBlock(
        alpha: Double,
        packedA: DoubleArray,
        aOffset: Int,
        aGroupStride: Int,
        packedB: DoubleArray,
        bOffset: Int,
        bGroupStride: Int,
        rows: Int,
        columns: Int,
        depth: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
    ): Unit = unsupportedEngineOperation(name, "productBlock")

    /**
     * How many right-hand sides this engine recommends substituting at once, for a block of [order] steps
     * over [sides] of them.
     *
     * Always at least one and never more than [sides] when that is positive. It is a grouping and not a lane
     * count, and it is independent of the register tile, the panel grouping and the diagonal block the
     * scheduling chose: a diagonal block is as wide as the dependency chain is long, and this is how much of
     * the work beside it is done together.
     */
    public fun rightHandSideGroup(order: Int, sides: Int): Int = unsupportedEngineOperation(name, "rightHandSideGroup")

    /**
     * Whether a group of right-hand sides this far apart is worth copying into adjacent storage first.
     *
     * The copy is two passes over a block of the caller's data, out and back, and it buys a substitution
     * whose independent values lie next to each other. An engine whose arithmetic is a value at a time gains
     * nothing from that and answers false, and so does one whose right-hand sides are already adjacent.
     * Where this answers true the caller gathers.
     */
    public fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean = unsupportedEngineOperation(
        name,
        "gathersRightHandSides",
    )

    /**
     * Solves `M · X = X` in place over one diagonal block, for every right-hand side at once.
     *
     * Substitutes forward when [lower] and backward when it is not, dividing each finished step by its
     * diagonal coefficient unless [unitDiag]. No singularity check or particular result for singular or
     * non-finite inputs is promised.
     */
    public fun diagonalSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int = 0,
    ): Unit = unsupportedEngineOperation(name, "diagonalSolve")

    /**
     * Multiplies `X = M · X` in place over one diagonal block, for every right-hand side at once.
     *
     * The order is the one that leaves no step needed after it has been overwritten, which runs opposite to
     * the solve's: a lower block is substituted from its last step upward, because a step is the sum of the
     * steps at or before it and those must still hold what they arrived with.
     */
    public fun diagonalMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int = 0,
    ): Unit = unsupportedEngineOperation(name, "diagonalMultiply")

    /** Whole matrix vector product over validated operands. The caller validates and stages aliases before dispatch. */
    public fun computeGemv(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ): Unit = unsupportedEngineOperation(name, "computeGemv")

    /** Whole symmetric matrix vector product over the stored triangle. The caller validates and stages aliases before dispatch. */
    public fun computeSymv(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ): Unit = unsupportedEngineOperation(name, "computeSymv")

    /** Whole rank one update over validated operands. The caller validates and stages aliases before dispatch. */
    public fun computeGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion): Unit =
        unsupportedEngineOperation(
            name,
            "computeGer",
        )

    /** Whole symmetric rank one update over the stored triangle. The caller validates and stages aliases before dispatch. */
    public fun computeSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure): Unit =
        unsupportedEngineOperation(name, "computeSyr")

    /** Whole symmetric rank two update over the stored triangle. The caller validates and stages aliases before dispatch. */
    public fun computeSyr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
    ): Unit = unsupportedEngineOperation(name, "computeSyr2")

    /** Whole triangular vector substitution over validated operands. The caller validates and stages aliases before dispatch. */
    public fun computeTrsv(
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
    ): Unit = unsupportedEngineOperation(name, "computeTrsv")

    /** Whole triangular vector product over validated operands. The caller validates and stages aliases before dispatch. */
    public fun computeTrmv(
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
    ): Unit = unsupportedEngineOperation(name, "computeTrmv")

    /** Whole dense matrix product over validated operands. The caller validates and stages aliases before dispatch. */
    public fun computeGemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
    ): Unit = unsupportedEngineOperation(name, "computeGemm")

    /** Whole symmetric matrix product over the stored triangle. The caller validates and stages aliases before dispatch. */
    public fun computeSymm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        rightSide: Boolean = false,
    ): Unit = unsupportedEngineOperation(name, "computeSymm")

    /** Whole symmetric rank k update over the stored triangle. The caller validates and stages aliases before dispatch. */
    public fun computeSyrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ): Unit = unsupportedEngineOperation(name, "computeSyrk")

    /** Whole symmetric rank two k update over the stored triangle. The caller validates and stages aliases before dispatch. */
    public fun computeSyr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ): Unit = unsupportedEngineOperation(name, "computeSyr2k")

    /** Whole triangular matrix product over validated operands. The caller validates and stages aliases before dispatch. */
    public fun computeTrmm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean = false,
    ): Unit = unsupportedEngineOperation(name, "computeTrmm")

    /** Whole triangular matrix substitution over validated operands. The caller validates and stages aliases before dispatch. */
    public fun computeTrsm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean = false,
    ): Unit = unsupportedEngineOperation(name, "computeTrsm")

    /** Whole dense product restricted to the selected output triangle. The caller validates and stages aliases before dispatch. */
    public fun computeGemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ): Unit = unsupportedEngineOperation(name, "computeGemmt")

    /** Explicit composition without changing the platform default. */
    public companion object {
        /**
         * Combines available arithmetic with ordered routing preferences and complete portable coverage.
         *
         * Backends named by no applicable rule contribute no arithmetic. Known unavailable built-in backends fall through to
         * later preferences and then portable Kotlin; unknown source identifiers are rejected. This never changes [koblas]. The caller chooses an
         * applicable profile explicitly; automatic platform selection also checks [EngineProfile.appliesTo].
         * Panel, product and triangular families are selected unconditionally before a schedule starts,
         * because packing and grouping must use the same geometry as the selected arithmetic.
         * Whole-call portable preferences retain the shared schedule with those separately selected families.
         */
        public fun combine(profile: EngineProfile, vararg backends: KoblasEngine): KoblasEngine {
            require(profile.rules.none { rule -> !rule.unconditional && rule.implementations.any { it.family } }) {
                "Scheduled kernel families require unconditional selection before packing"
            }
            return CombinedEngine(CompiledRouting(profile, backends.toList()), profile.name)
        }
    }
}

private fun unsupportedEngineOperation(name: String, operation: String): Nothing =
    throw UnsupportedOperationException("$operation is not supplied by $name")
