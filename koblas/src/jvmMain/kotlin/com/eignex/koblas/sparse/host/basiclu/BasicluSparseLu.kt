package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.host.F64SparseLuAdapter
import com.eignex.koblas.withColumn

/** Sparse LU and Forrest-Tomlin basis updates backed by a host BASICLU. */
public class BasicluSparseLu(
    /** Policy for this backend instance. */
    public val config: BasicluConfig = BasicluConfig(),
) : F64SparseLuAdapter(config.factorizeMin) {
    private val calls = BasicluCalls(config)

    override val name: String get() = BackendNames.BASICLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
    override val nativeAvailable: Boolean get() = calls.available

    override val supportsBasisUpdates: Boolean get() = nativeAvailable

    /** BASICLU offers no row scaling, so an equilibrated factorization stays with the portable one. */
    override fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization {
        if (equilibrate) return F64ReferenceSparseLinearAlgebra.factor(a, equilibrate = true)
        val target = factored(a) ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        return BasicluFactorization(target, calls)
    }

    /**
     * A basis is factorized natively at any size, since what a caller wants from BASICLU is the update that
     * follows and the gate the general factorization answers to has nothing to say about it.
     */
    override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization {
        requireSquare(basis, "factorBasis")
        if (!nativeAvailable) return super.factorBasis(basis)
        val target = factored(basis) ?: return F64SingularBasisFactorization(this, basis)
        return BasicluBasisFactorization(this, basis, target, calls)
    }

    /** An initialized object carrying [a]'s factors, or null when BASICLU would not factor it. */
    private fun factored(a: F64SparseMatrix): BasicluObject? {
        val target = calls.create(a.rows) ?: return null
        val status = calls.factorize(target, a.copyColumnPointers(), a.copyRowIndices(), a.values)
        if (status == BasicluStatus.OK) return target
        calls.free(target)
        target.arena.close()
        return null
    }
}

/** What [BasicluSparseLu.factorBasis] answers with for a basis BASICLU called singular. */
private class F64SingularBasisFactorization(private val owner: BasicluSparseLu, override val basis: F64SparseMatrix) :
    F64BasisFactorization {
    override val n: Int get() = basis.rows
    override val failedAt: Int get() = SINGULAR_POSITION_UNKNOWN
    override val nnz: Int get() = 0
    override val rcond: Double get() = 0.0

    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization =
        owner.factorBasis(basis.withColumn(column, entering))

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        throw SingularMatrix(failedAt, "solve: the factorization is singular")
}
