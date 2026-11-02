@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.internal.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/**
 * The logical panel shapes a matrix algorithm asks a backend for.
 *
 * Each names work, not an instruction: how many values are reduced or updated together is the backend's
 * answer to [KoblasEngine.executionGroup], and a caller that hard-coded four columns would be writing
 * one machine's geometry into an algorithm every machine runs.
 */
internal enum class PanelWork {
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
     * Two questions carry this name. [KoblasEngine.executionGroup] answers how many right-hand sides a
     * sparse traversal should hand over at a time, where `columns` is how many there are to choose from.
     * [KoblasEngine.indexedRankUpdate] and [KoblasEngine.indexedCoupledUpdate] are the arithmetic
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
     * array the caller supplied. [KoblasEngine.indexedColumnUpdate] is the arithmetic over a group of
     * this shape.
     */
    SparseRightHandSideReduction,
}
