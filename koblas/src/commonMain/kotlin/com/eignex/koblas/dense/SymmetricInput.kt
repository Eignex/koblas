package com.eignex.koblas.dense

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.requireSquare
import com.eignex.koblas.transpose

/** Return a matrix whose lower triangle is the selected symmetric input. */
internal fun F64DenseMatrix.asLowerSymmetricInput(lower: Boolean, operation: String): F64DenseMatrix {
    requireSquare(this, operation)
    return if (lower) this else transpose()
}
