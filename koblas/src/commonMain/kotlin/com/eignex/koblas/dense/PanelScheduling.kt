package com.eignex.koblas.dense

/**
 * Visits [columns] logical columns in groups of at most [group], which is what a backend recommended.
 *
 * The traversal is the same whatever the recommendation is. A group wider than what is left becomes the
 * remainder, a recommendation of one is a column at a time, and a backend that answered with nothing
 * positive still gets a correct call rather than a loop that never advances.
 *
 * One traversal for every grouping a call makes, whether what is grouped is an operand's columns or the
 * right-hand sides beside a triangle or a sparse column, and walked by the route that describes a call as
 * well as by the call itself. A route walking its own copy would name windows the call never cut, and the
 * last group of an extent that does not divide is what such a copy misses: it is shorter than the rest and
 * may reach a different body.
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
