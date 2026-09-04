package com.eignex.koblas.sparse.host

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT

/*
 * Copying a native array out into one of koblas's own, which every SuiteSparse binding does on the way back
 * from an extraction. An empty read copies nothing rather than asking the segment for zero elements, since
 * a library with nothing to hand back may leave the pointer null.
 */

/** [count] ints from [segment], which is already sized to hold them. */
internal fun readInts(segment: MemorySegment, count: Int): IntArray {
    val out = IntArray(count)
    if (count > 0) MemorySegment.copy(segment, JAVA_INT, 0L, out, 0, count)
    return out
}

/** [count] doubles from [segment], which is already sized to hold them. */
internal fun readDoubles(segment: MemorySegment, count: Int): DoubleArray {
    val out = DoubleArray(count)
    if (count > 0) MemorySegment.copy(segment, JAVA_DOUBLE, 0L, out, 0, count)
    return out
}
