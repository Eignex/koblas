package com.eignex.koblas.sparse.host.cholmod

/**
 * `L` in CSC and the fill-reducing permutation, copied out of a converted factor.
 *
 * The diagonal is first in each column, which is where an `L·D·Lᵀ` caller finds `D`.
 */
internal class CholmodFactors(
    val order: Int,
    val colPtr: IntArray,
    val rowIdx: IntArray,
    val values: DoubleArray,
    val permutation: IntArray,
)
