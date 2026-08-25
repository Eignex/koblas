package com.eignex.koblas.dense

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.requireSquare
import com.eignex.koblas.transpose

/**
 * Return a matrix whose lower triangle is the symmetric input selected by [uplo]. A full input is checked;
 * an upper input is transposed once so the lower-only LAPACK seam can consume it.
 */
internal fun F64DenseMatrix.asLowerSymmetricInput(uplo: Uplo, operation: String): F64DenseMatrix {
    requireSquare(this, operation)
    return when (uplo) {
        Uplo.LOWER -> this

        Uplo.UPPER -> transpose()

        Uplo.FULL -> {
            for (j in 0 until cols) {
                for (i in j + 1 until rows) {
                    require(this[i, j] == this[j, i]) {
                        "$operation requires a symmetric matrix with uplo=FULL; " +
                            "A($i, $j)=${this[i, j]} but A($j, $i)=${this[j, i]}"
                    }
                }
            }
            this
        }
    }
}
