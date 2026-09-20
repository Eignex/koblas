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
