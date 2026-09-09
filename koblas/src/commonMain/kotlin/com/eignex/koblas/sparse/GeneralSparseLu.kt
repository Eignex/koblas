@file:Suppress("UndocumentedPublicFunction")

package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SparseMatrix

/** General sparse LU supplied by HFactor. */
public interface GeneralSparseLu : Backend {
    /** Factorizes the square [a] for general solves. */
    public fun factor(a: SparseMatrix): SparseLuFactorization
}
