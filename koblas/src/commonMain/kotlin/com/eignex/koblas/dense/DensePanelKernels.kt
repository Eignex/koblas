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
     * Dense right-hand sides visited together while a sparse operand's indices and values are walked once.
     *
     * The arithmetic belongs to the sparse traversal, which reads indices this contract knows nothing about,
     * so there is no kernel here for it. What a sparse caller takes from this contract is the grouping, so
     * that its right-hand-side width is the local backend's choice rather than a constant of its own.
     */
    SparseRightHandSides,
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
 * Every window a call writes must be disjoint from every source window it reads, and the two windows a
 * coupled pass writes must be disjoint from each other. A destination read as part of writing it is not a
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
     */
    public fun executionGroup(work: PanelWork, rows: Int, columns: Int): Int

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
}
