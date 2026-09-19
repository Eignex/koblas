package com.eignex.koblas.dense

/**
 * Visits [columns] logical columns in groups of at most [group], which is what a backend recommended.
 *
 * The traversal is the same whatever the recommendation is. A group wider than what is left becomes the
 * remainder, a recommendation of one is a column at a time, and a backend that answered with nothing
 * positive still gets a correct call rather than a loop that never advances.
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
