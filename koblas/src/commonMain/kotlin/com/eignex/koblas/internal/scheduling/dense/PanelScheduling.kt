package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.KoblasEngine

/**
 * Visits [columns] in groups of at most [group], clamped to at least one.
 */
internal inline fun forEachPanel(columns: Int, group: Int, action: (start: Int, width: Int) -> Unit) {
    val step = if (group < 1) 1 else group
    var start = 0
    while (start < columns) {
        val width = if (columns - start < step) columns - start else step
        action(start, width)
        start += width
    }
}

/**
 * The window each group of a selected-triangle traversal shares, beside the corner its own columns take.
 *
 * [fromDiagonal] says whether the last column's own diagonal entry belongs to that window: a rank update
 * writes it, while a symmetric product has already read it in the corner.
 */
internal inline fun forEachTrianglePanel(
    n: Int,
    group: Int,
    lower: Boolean,
    fromDiagonal: Boolean,
    action: (start: Int, width: Int, window: Int, rows: Int) -> Unit,
) {
    val diagonal = if (fromDiagonal) 1 else 0
    forEachPanel(n, group) { start, width ->
        val window = if (lower) start + width - diagonal else 0
        action(start, width, window, if (lower) n - window else start + diagonal)
    }
}

/**
 * Steps of the order a triangular vector routine finishes column by column before the rest of the call sees
 * them through one panel.
 *
 * The scheduling's own number, separate from [TRIANGULAR_DIAGONAL_BLOCK] because here a block has one
 * right-hand side. Inside a block the substitution is one [KoblasEngine] diagonal call, a dependency
 * chain nothing overlaps; between blocks the whole block is a single panel as wide as the block, which a
 * backend can keep its destination in registers across. A larger block moves work into the chain, a smaller
 * one makes each panel narrower.
 */
internal const val TRIANGULAR_VECTOR_BLOCK: Int = 64
