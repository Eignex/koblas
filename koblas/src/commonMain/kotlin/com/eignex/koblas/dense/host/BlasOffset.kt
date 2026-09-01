package com.eignex.koblas.dense.host

/**
 * The array index CBLAS/LAPACKE reach at [offset] with [stride] over [len] elements: [offset] itself for a
 * non-negative stride, since BLAS then walks forward from there, or the last element the negative stride
 * reaches, since BLAS treats [offset] as one past that end. The multiplication runs in `Long` first, so a
 * huge stride/len combination cannot overflow the intermediate result before it lands back in a valid index.
 */
internal fun blasOffset(offset: Int, stride: Int, len: Int): Int =
    if (stride < 0) (offset.toLong() + (len - 1).toLong() * stride).toInt() else offset
