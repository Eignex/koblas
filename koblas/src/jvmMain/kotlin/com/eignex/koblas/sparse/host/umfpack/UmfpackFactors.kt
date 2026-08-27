package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.core.F64SparseMatrix

/** What UMFPACK's extraction hands back, already in CSC with the scaling normalised to a multiplier. */
internal class UmfpackFactors(
    val lower: F64SparseMatrix,
    val upper: F64SparseMatrix,
    val rowOrder: IntArray,
    val columnOrder: IntArray,
    val rowScaling: DoubleArray,
)
