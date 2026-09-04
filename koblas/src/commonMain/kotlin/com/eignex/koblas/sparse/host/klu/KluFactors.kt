package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.core.F64SparseMatrix

/** What KLU's extraction hands back, in CSC with the scaling normalised to a multiplier. */
internal class KluFactors(
    val lower: F64SparseMatrix,
    val upper: F64SparseMatrix,
    val offDiagonal: F64SparseMatrix,
    val rowOrder: IntArray,
    val columnOrder: IntArray,
    val rowScaling: DoubleArray,
)
