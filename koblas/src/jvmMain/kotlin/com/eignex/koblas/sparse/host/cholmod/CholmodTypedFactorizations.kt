package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseCholeskyFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLdlFactorization

/*
 * CHOLMOD answers both symmetric routines with one factor object, but the two are different factorizations
 * and only one of them has a `D`. Rather than one class claiming both interfaces and lying about the field it
 * does not have, each routine wraps the shared factorization in the type its own contract names.
 */

/** CHOLMOD's `A = L·Lᵀ`, whose `L` carries the real diagonal. */
public class CholmodCholeskyFactorization internal constructor(
    private val base: CholmodFactorization,
) : F64SparseCholeskyFactorization, F64SparseFactorization by base {
    override val l: F64SparseMatrix get() = base.lowerFactor

    override val order: IntArray get() = base.ordering

    override fun close(): Unit = base.close()
}

/** CHOLMOD's `A = L·D·Lᵀ`, whose diagonal is split out of the factor it stores it in. */
public class CholmodLdlFactorization internal constructor(
    private val base: CholmodFactorization,
) : F64SparseLdlFactorization, F64SparseFactorization by base {
    override val l: F64SparseMatrix get() = base.lowerFactor

    override val d: DoubleArray get() = base.diagonalFactor

    override val order: IntArray get() = base.ordering

    override fun close(): Unit = base.close()
}
