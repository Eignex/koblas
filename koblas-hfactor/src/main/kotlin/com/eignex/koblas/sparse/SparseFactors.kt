@file:Suppress("UndocumentedPublicProperty")

package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix

/** Raised because HFactor keeps an updateable basis representation rather than materialized factors. */
public class FactorsNotExposed(factor: String) :
    UnsupportedOperationException("this factorization does not expose $factor")

/** HFactor's retained general sparse LU result. */
public interface SparseLuFactorization : SparseFactorization {
    public val l: SparseMatrix get() = throw FactorsNotExposed("l")
    public val u: SparseMatrix get() = throw FactorsNotExposed("u")
    public val rowOrder: IntArray get() = throw FactorsNotExposed("rowOrder")
    public val columnOrder: IntArray get() = throw FactorsNotExposed("columnOrder")
    public val rowScaling: DoubleArray get() = throw FactorsNotExposed("rowScaling")
    public val offDiagonal: SparseMatrix
        get() = SparseMatrix.wrap(n, n, IntArray(n + 1), IntArray(0), DoubleArray(0))
}
