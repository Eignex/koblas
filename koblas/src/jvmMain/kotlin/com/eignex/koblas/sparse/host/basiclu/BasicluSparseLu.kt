package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter
import com.eignex.koblas.sparse.host.f64EquilibrationScale
import com.eignex.koblas.sparse.host.f64ScaledValues
import com.eignex.koblas.withColumn

/**
 * Sparse LU and Forrest-Tomlin basis updates backed by a host BASICLU.
 *
 * Open so a bundled provider is one of these under another name rather than a wrapper around one. What a
 * caller reaches this backend by name for is the routines it carries outside the seam, and a wrapper
 * would hide their type. Only the name and the ranking are meant to vary.
 */
public open class BasicluSparseLu(
    /** Policy for this backend instance. */
    public val config: BasicluConfig = BasicluConfig(),
) : F64SparseDecompositionsAdapter(
    config.factorizeMin,
    config.equilibrate,
    BackendMetadata(options = config.options.metadataOptions()),
),
    F64GeneralSparseLu,
    F64BasisFactorizations {
    private val calls = BasicluCalls(config)

    override val name: String get() = BackendNames.BASICLU
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
    final override val nativeAvailable: Boolean get() = calls.available

    /**
     * Whether [factorBasis] answers with a factorization that updates its factors in place. False without
     * the library, where a replacement costs a factorization and a caller pacing its own refactorizations
     * has nothing left to pace.
     */
    public val supportsBasisUpdates: Boolean get() = nativeAvailable

    /**
     * BASICLU offers no row scaling of its own, so equilibration is applied to the values handed over and
     * undone in the solves, by the same power-of-two factors the portable factorization uses.
     */
    final override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
        val rowIdx = a.copyRowIndices()
        val scale = if (equilibrate) f64EquilibrationScale(a.rows, rowIdx, a.values) else null
        val values = if (scale == null) a.values else f64ScaledValues(rowIdx, a.values, scale)
        val target = factored(a, rowIdx, values)
            ?: return F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
        return BasicluFactorization(target, calls, scale)
    }

    /**
     * Factor a simplex [basis] for column replacements, which may bring in any column at all.
     *
     * BASICLU's own routine rather than a seam method: it carries the basis independently of any matrix,
     * which is what lets an entering column be arbitrary, and no other sparse LU koblas binds does that.
     * A basis is factorized natively at any size, since what a caller wants from BASICLU is the update that
     * follows and the gate the general factorization answers to has nothing to say about it.
     */
    final override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization {
        requireSquare(basis, "factorBasis")
        if (!nativeAvailable) return F64RefactoringBasisFactorization(this, basis, factor(basis))
        val target = factored(basis, basis.copyRowIndices(), basis.values)
            ?: return F64SingularBasisFactorization(this, basis)
        return BasicluBasisFactorization(this, basis, target, calls)
    }

    /**
     * An initialized object carrying the factors of [a]'s pattern with [values], or null when BASICLU would
     * not factor it. The values come in separately because an equilibrated factorization scales them first.
     */
    private fun factored(a: F64SparseMatrix, rowIdx: IntArray, values: DoubleArray): BasicluObject? {
        val target = calls.create(a.rows) ?: return null
        val status = calls.factorize(target, a.copyColumnPointers(), rowIdx, values)
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
