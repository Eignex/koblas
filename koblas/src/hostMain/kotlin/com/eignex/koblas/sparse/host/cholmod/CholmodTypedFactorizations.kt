package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64QuasiDefiniteLdlFactorization
import com.eignex.koblas.sparse.F64SparseCholeskyFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.FactorizationInertia

/*
 * CHOLMOD answers both symmetric routines with one factor object, but the two are different factorizations
 * and only one of them has a `D`. Each routine wraps the shared factorization in the type its own contract
 * names rather than one class claiming both and lying about the field it does not have.
 */

/** CHOLMOD's `A = L·Lᵀ`, whose `L` carries the real diagonal. */
public class CholmodCholeskyFactorization internal constructor(private val base: CholmodFactorization) :
    F64SparseCholeskyFactorization,
    F64SparseFactorization by base {
    override val l: F64SparseMatrix get() = base.lowerFactor

    override val order: IntArray get() = base.ordering

    override fun close(): Unit = base.close()
}

/** CHOLMOD's `A = L·D·Lᵀ`, whose diagonal is split out of the factor it stores it in. */
public class CholmodLdlFactorization internal constructor(private val base: CholmodFactorization) :
    F64QuasiDefiniteLdlFactorization,
    F64SparseFactorization by base {
    override val l: F64SparseMatrix get() = base.lowerFactor

    override val d: DoubleArray get() = base.diagonalFactor

    override val inertia: FactorizationInertia
        get() = d.inertia()

    override val order: IntArray get() = base.ordering

    override fun close(): Unit = base.close()
}

private fun DoubleArray.inertia(): FactorizationInertia = FactorizationInertia(
    positive = count { it > 0.0 },
    negative = count { it < 0.0 },
    zero = count { it == 0.0 },
)
