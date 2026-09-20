package com.eignex.koblas.dense

/**
 * Visits [columns] in groups of at most [group], clamped to at least one.
 * Shared by dispatch and route reporting so both see the same short final group.
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
 * Shared by dispatch and route reporting, so the lengths a route names are the ones the call hands over.
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
 * The stored off-diagonal part of every column of a triangle, in the order the dependence reaches them.
 *
 * One walk for all four variants of a triangular vector routine, because the window is the same in each and
 * only the order differs, which the caller states through [ascending].
 */
internal inline fun forEachTriangularColumn(
    n: Int,
    lower: Boolean,
    ascending: Boolean,
    action: (j: Int, window: Int, rows: Int) -> Unit,
) {
    for (step in 0 until n) {
        val j = if (ascending) step else n - 1 - step
        action(j, if (lower) j + 1 else 0, if (lower) n - 1 - j else j)
    }
}
