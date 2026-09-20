@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.dense

/**
 * The logical panel shapes a matrix algorithm asks a backend for.
 *
 * Each names work, not an instruction: how many values are reduced or updated together is the backend's
 * answer to [DensePanelKernels.executionGroup], and a caller that hard-coded four columns would be writing
 * one machine's geometry into an algorithm every machine runs.
 */
public enum class PanelWork {
    /** Several logical columns reduced against one vector, producing one value per column. */
    MultiDot,

    /** Several logical columns accumulated with their coefficients into one destination window. */
    ColumnUpdate,

    /** One pass that both updates a destination window and reduces the same columns against a vector. */
    CoupledDotUpdate,

    /** One source window accumulated with its coefficients into several destination columns. */
    RankUpdate,

    /**
     * Dense right-hand sides visited together while a sparse operand's indices and values are walked once,
     * where the panel's destination is the dense block.
     *
     * Two questions carry this name. [DensePanelKernels.executionGroup] answers how many right-hand sides a
     * sparse traversal should hand over at a time, where `columns` is how many there are to choose from.
     * [DensePanelKernels.indexedRankUpdate] and [DensePanelKernels.indexedCoupledUpdate] are the arithmetic
     * over one such group, where the group is the panel's `rows` and the stored entries it walks are its
     * `columns`, so a backend that vectorises does it across the right-hand sides and never across the
     * indices.
     */
    SparseRightHandSides,

    /**
     * The same panel with the destination the other way round: one accumulator per right-hand side, which a
     * column of a transposed sparse operand reduces into.
     *
     * Its own entry because a backend may want a different amount of work in it. What the two shapes do
     * with a group differs: an update spreads a coefficient over a destination window, where a wider group
     * spends one walk of the column's indices on more of it; a reduction carries an accumulator per
     * right-hand side, which a group of one can keep in a register and a wider group has to keep in the
     * array the caller supplied. [DensePanelKernels.indexedColumnUpdate] is the arithmetic over a group of
     * this shape.
     */
    SparseRightHandSideReduction,
}

/**
 * Panel arithmetic over a column-major window, with the execution grouping chosen locally.
 *
 * A panel is `rows` by `columns` starting at an offset, each column `lda` apart, and every entry of a column
 * adjacent. Those extents are the operation's, valid at any positive size, and an implementation handles a
 * panel narrower than its own grouping or shorter than a vector by doing the arithmetic some other way rather
 * than by refusing it. Callers ask [executionGroup] how much to hand over at a time and loop; passing more or
 * less than it recommended is still correct, which is what makes the recommendation a recommendation.
 *
 * Every product is formed, including one whose coefficient is zero. This is matrix arithmetic, where a
 * position the traversal reaches contributes whatever it evaluates to, and a zero coefficient against an
 * infinity is a NaN that belongs in the result. Level 1 [DenseVectorKernels.axpy] is specified the other way
 * round and returns without touching its destination, so a matrix update cannot be handed to it unchanged.
 *
 * Results agree with the direct definition to within rounding rather than bit for bit: an implementation may
 * fuse a multiply and an add, and may reduce a column in lanes and combine them as a tree.
 *
 * Empty extents are legal and settled here rather than left to where a loop happens to sit. A panel with no
 * columns does nothing and reads nothing. A panel with no rows reads no matrix, no shared vector and no
 * coefficient: the three whose destination is the row window write nothing at all, and [multiDot] still
 * writes its own outputs, which the columns select rather than the rows, so with a nonzero `beta` it reads
 * those outputs and scales them. A caller may therefore pass whatever it likes for a window with no extent,
 * and the guards below are what that promise rests on.
 *
 * Implementations read and write only the windows their offsets and extents select, validate nothing, and
 * allocate nothing. A caller that needs several results supplies the array they are written into.
 *
 * Every window a call writes must be disjoint from every source window it reads. The two windows a coupled
 * pass writes must also be disjoint, except for the overlap between corresponding right-hand sides explicitly
 * supported by [indexedCoupledUpdate]. A destination read as part of writing it is not a
 * source: accumulating into the window already there is what most of these do, and a nonzero `beta` reading
 * the output it scales is the same thing. Disjoint windows of one backing array are fine, and so is one
 * source overlapping another: a rank update over a single vector passes the same window twice on purpose.
 * What is not supported is a source overlapping a destination, such as a rank update whose vector lies
 * inside the panel it writes. These leaves check nothing; a public matrix call stages an operand that would
 * break the rule before it reaches one, which is where that guarantee is kept.
 */
public interface DensePanelKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /**
     * How many columns of [work] this backend recommends processing together, for a panel of [rows] by
     * [columns].
     *
     * Always at least one and never more than [columns] when [columns] is positive. The answer is a local
     * tuning choice made from the resolved vector width and small measured rules, and it is not a lane count:
     * a group of four columns and four lanes are different numbers that happen to agree on some machines.
     *
     * [contiguous] is part of the question for the same reason it is part of [implementationFor]'s: a backend
     * whose body over adjacent entries is a different body may want a different amount of work in it. A
     * caller that will hand over a strided window has to say so, or it is given a recommendation for a body
     * it is not going to reach. A scheduling of its own may then take less than what was recommended, for a
     * cache or a workspace it is keeping within; passing less is always correct.
     */
    public fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean = true): Int

    /**
     * Whether a panel of these extents is worth copying into adjacent rows before it is handed over.
     *
     * A caller that can copy a strided window into adjacent storage needs to know whether the copy buys
     * anything before it pays for it, and the answer is the backend's, because only the backend knows what
     * its body does with the two layouts. A portable body reads one entry at a time either way and says no
     * at every extent. A vector body says yes where the panel is wide enough that the arithmetic it gains
     * outweighs a pass over the data, which is not the same width as the one where its body starts running:
     * one lane block is a single vector operation per column, and a copy is not repaid by that.
     *
     * This is a recommendation about copying rather than a claim about which body runs, and the two are
     * asked separately. Where it is true, [implementationFor] also names a different body for the two
     * layouts, so that a route reports the body the copy was made for; where it is false the body may still
     * differ, and the caller has simply been told not to pay for the difference.
     */
    public fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean

    /**
     * The arithmetic implementation a panel of this [work] and these extents reaches.
     *
     * [name] identifies a selection, which is a different question. A backend whose vector body needs a full
     * lane block answers with the portable one for a panel shorter than that, and a caller reporting the
     * selection name would have named something that did not run.
     *
     * [contiguous] is part of the question rather than a detail of it: a vector body loads a lane block from
     * consecutive entries, so a panel whose shared vector is strided is scalar work at any width.
     */
    public fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean = true): String

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
    )

    /**
     * `y(i) += Σ (alpha · x(c)) · a(i, c)` over [rows] destination entries and [columns] panel columns.
     *
     * [alpha] scales each column's coefficient rather than the finished sum, so a caller splitting a long
     * panel into groups gets the same arithmetic whatever grouping it chose.
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
    )

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
    )

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
    )

    /**
     * `y(i) += Σ (alpha · values(c)) · a(i, indices(c))` over a panel whose columns an index array selects.
     *
     * The sparse counterpart of [columnUpdate], and the shape a sparse column takes when its dense operand
     * carries several right-hand sides. [rows] is the group of right-hand sides, which is the axis with no
     * dependences between its entries and therefore the one a backend may vectorise; [columns] is how many
     * stored entries the column contributes, read as `indices` and `values` from [fromIndex] onward.
     *
     * An entry of the panel is at `aOffset + i · rowStride + indices(c) · indexStride`, which is two strides
     * rather than a leading dimension because either axis may be the contiguous one. A dense block in
     * column-major storage has its right-hand sides a leading dimension apart, and one staged into
     * right-hand-side-major order has them adjacent; the arithmetic is the same and only the strides differ.
     * [implementationFor] answers which body a given panel reaches, and a caller that wants the vector one
     * has to make [rowStride] one for it.
     *
     * The destination is [rows] adjacent entries from [yOffset] and is accumulated into. Every product is
     * formed, as everywhere in this contract: a stored zero of a sparse operand is a position its pattern
     * reaches, so a zero against an infinity is a NaN the caller asked for.
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
    )

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
     * extents in play. A backend may hold several entries in registers at once, so two logical entries
     * sharing an address would lose an update rather than apply both. This is the one precondition these two
     * leaves do not share: a reduction may read a position twice, a scatter may not write one twice.
     *
     * Nothing here is validated, as nowhere in this contract is. A public matrix call passes windows of its
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
    )

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
     * column into a scratch first would be a pass over it per column of the sparse operand, which is what
     * this shape exists to avoid.
     *
     * [a] and [b] are windows of the same shape addressed alike, from [offset] with the two strides
     * [indexedColumnUpdate] describes; a symmetric product passes its destination and its source, which are
     * different arrays. The reduction lands in [rows] entries from [sumOffset] spaced [rowStride] apart.
     * This window may coincide with one indexed column of [a], with each reduction entry overlapping only
     * the scatter entry for the same logical right-hand side. An overlap between different right-hand sides
     * is unsupported. Where both updates target the same entry, the scatter is applied before the reduction
     * for each stored position. All source windows remain disjoint from both destination windows.
     *
     * [excluded] is a position relative to this run, in `0 until columns`, whose product is scattered but
     * not reduced; `-1` excludes none. The column of [b] it selects is not read, though [pivot] still is,
     * because the scatter runs at that position like any other. A symmetric column excludes its own
     * diagonal because that contribution is already scattered into the pivot row; the exclusion suppresses
     * that second contribution, while the scatter still writes the overlapping destination.
     *
     * Every product is formed, as everywhere in this contract, and the reduction is accumulated into rather
     * than written, so its terms reach the destination in the order the run is walked.
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
    )
}
