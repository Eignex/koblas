package com.eignex.koblas.internal.dense

/** `M(p, q)` for a diagonal block starting at [start], which is the entry a substitution reads. */
internal fun triangleEntry(
    t: DoubleArray,
    n: Int,
    start: Int,
    p: Int,
    q: Int,
    transposed: Boolean,
    offset: Int = 0,
): Double = if (transposed) {
    t[offset + start + q + (start + p) * n]
} else {
    t[offset + start + p + (start + q) * n]
}
