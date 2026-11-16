package com.eignex.koblas.internal.storage.dense

/** First selected row in a column, clipped to the logical row extent. Null selects the full column. */
internal fun selectedRowStart(column: Int, rows: Int, lower: Boolean?): Int =
    if (lower == true) minOf(column, rows) else 0

/** Exclusive end of selected rows, clipped to the logical row extent. */
internal fun selectedRowEnd(column: Int, rows: Int, lower: Boolean?): Int =
    if (lower == false) minOf(column, rows - 1) + 1 else rows
