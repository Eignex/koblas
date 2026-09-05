package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.factorization.cholesky.F64SparseUpLookingCholesky
import com.eignex.koblas.sparse.factorization.ldl.F64QuasiDefiniteUpLookingLdl
import com.eignex.koblas.sparse.factorization.lu.F64SparseMarkowitzLu
import com.eignex.koblas.sparse.factorization.lu.NO_DROP
import com.eignex.koblas.sparse.factorization.qr.F64SparseHouseholderQr

/**
 * The portable sparse LU, with the two knobs only it has.
 *
 * Both are policy of the factorization rather than of one call, so they are settled here the way a host
 * binding settles its own in its constructor. [Default] is the policy a caller who has not thought about
 * either wants, and is what the seam falls back to.
 *
 * @property equilibrate scale rows by a power of two before factorizing; the solves undo it.
 * @property dropTolerance discard produced entries this far below the largest magnitude, giving an
 *   incomplete factorization.
 */
public open class F64ReferenceSparseDecompositions(
    public val equilibrate: Boolean = false,
    public val dropTolerance: Double = NO_DROP,
) : F64SparseDecompositions,
    F64GeneralSparseLu,
    F64SparseCholesky,
    F64QuasiDefiniteLdl,
    F64SparseQr {
    init {
        require(dropTolerance >= 0.0) { "dropTolerance must not be negative" }
    }

    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    override fun factor(a: F64SparseMatrix): F64SparseLuFactorization =
        F64SparseMarkowitzLu.factorCsc(a, equilibrate, dropTolerance)

    /** Neither knob reaches here: equilibration would break the symmetry, and a drop tolerance would leave
     *  an incomplete factor, which is a preconditioner rather than the factorization this promises. */
    override fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization = F64SparseUpLookingCholesky.factorLower(
        a,
    )

    /** The knobs stay out of this one too, for the reason above. */
    override fun quasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization =
        F64QuasiDefiniteUpLookingLdl.factorLower(a)

    /** Nor do they reach this one: equilibration would change the least-squares problem being solved. */
    override fun qr(a: F64SparseMatrix): F64SparseQrFactorization = F64SparseHouseholderQr.factor(a)

    /**
     * The elimination tree and the column counts of `L`, held so that refactorizing this pattern reaches the
     * numeric sweep directly. The transposed triangle the sweep reads carries values, so it is not kept.
     */
    override fun analyzeCholesky(a: F64SparseMatrix): F64SparseSymbolicAnalysis<F64SparseCholeskyFactorization> {
        val symbolic = F64SparseUpLookingCholesky.analyzeLower(a)
        return PatternOnlyAnalysis(a) { F64SparseUpLookingCholesky.factorLower(it, symbolic) }
    }

    /** The same structure the Cholesky analyzes, over a diagonal this one stores separately. */
    override fun analyzeQuasiDefiniteLdl(
        a: F64SparseMatrix,
    ): F64SparseSymbolicAnalysis<F64QuasiDefiniteLdlFactorization> {
        val symbolic = F64QuasiDefiniteUpLookingLdl.analyzeLower(a)
        return PatternOnlyAnalysis(a) { F64QuasiDefiniteUpLookingLdl.factorLower(it, symbolic) }
    }

    /** The column elimination tree, the row permutation, and both factors' entry counts. */
    override fun analyzeQr(a: F64SparseMatrix): F64SparseSymbolicAnalysis<F64SparseQrFactorization> {
        val symbolic = F64SparseHouseholderQr.analyze(a)
        return PatternOnlyAnalysis(a) { F64SparseHouseholderQr.factor(it, symbolic) }
    }

    /** No scaling and no drop tolerance, so `F64ReferenceSparseDecompositions.factor(a)` reads as the plain routine. */
    public companion object Default : F64ReferenceSparseDecompositions()
}
