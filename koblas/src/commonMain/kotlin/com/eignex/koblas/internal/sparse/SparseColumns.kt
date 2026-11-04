package com.eignex.koblas.internal.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.portable.sparse.forEachTriangleColumn

/** Pure column/support enumeration, without coefficient reads. */
@OptIn(UnsafeKoblasApi::class)
internal inline fun forEachSparseColumn(
    a: SparseMatrix,
    forward: Boolean = true,
    action: (column: Int, from: Int, to: Int) -> Unit,
) {
    forEachTriangleColumn(a.cols, forward) { column ->
        action(column, a.colPointers[column], a.colPointers[column + 1])
    }
}
