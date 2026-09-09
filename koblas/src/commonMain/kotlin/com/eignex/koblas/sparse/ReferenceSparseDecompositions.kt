package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.factorization.cholesky.SparseUpLookingCholesky
import com.eignex.koblas.sparse.factorization.ldl.QuasiDefiniteUpLookingLdl
import com.eignex.koblas.sparse.factorization.lu.NO_DROP
import com.eignex.koblas.sparse.factorization.lu.SparseMarkowitzLu
import com.eignex.koblas.sparse.factorization.qr.SparseHouseholderQr

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
public open class ReferenceSparseDecompositions(
    public val equilibrate: Boolean = false,
    public val dropTolerance: Double = NO_DROP,
) : SparseLapack,
    GeneralSparseLu,
    SparseCholesky,
    QuasiDefiniteLdl,
    SparseQr {
    init {
        require(dropTolerance >= 0.0) { "dropTolerance must not be negative" }
    }

    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    override fun factor(a: SparseMatrix): SparseLuFactorization =
        SparseMarkowitzLu.factorCsc(a, equilibrate, dropTolerance)

    /** Neither knob reaches here: equilibration would break the symmetry, and a drop tolerance would leave
     *  an incomplete factor, which is a preconditioner rather than the factorization this promises. */
    override fun cholesky(a: SparseMatrix): SparseCholeskyFactorization = SparseUpLookingCholesky.factorLower(
        a,
    )

    /** The knobs stay out of this one too, for the reason above. */
    override fun quasiDefiniteLdl(a: SparseMatrix): QuasiDefiniteLdlFactorization =
        QuasiDefiniteUpLookingLdl.factorLower(a)

    /** Nor do they reach this one: equilibration would change the least-squares problem being solved. */
    override fun qr(a: SparseMatrix): SparseQrFactorization = SparseHouseholderQr.factor(a)

    /**
     * The elimination tree and the column counts of `L`, held so that refactorizing this pattern reaches the
     * numeric sweep directly. The transposed triangle the sweep reads carries values, so it is not kept.
     */
    override fun analyzeCholesky(a: SparseMatrix): SparseSymbolicAnalysis<SparseCholeskyFactorization> {
        val symbolic = SparseUpLookingCholesky.analyzeLower(a)
        return PatternOnlyAnalysis(a) { SparseUpLookingCholesky.factorLower(it, symbolic) }
    }

    /** The same structure the Cholesky analyzes, over a diagonal this one stores separately. */
    override fun analyzeQuasiDefiniteLdl(a: SparseMatrix): SparseSymbolicAnalysis<QuasiDefiniteLdlFactorization> {
        val symbolic = QuasiDefiniteUpLookingLdl.analyzeLower(a)
        return PatternOnlyAnalysis(a) { QuasiDefiniteUpLookingLdl.factorLower(it, symbolic) }
    }

    /** The column elimination tree, the row permutation, and both factors' entry counts. */
    override fun analyzeQr(a: SparseMatrix): SparseSymbolicAnalysis<SparseQrFactorization> {
        val symbolic = SparseHouseholderQr.analyze(a)
        return PatternOnlyAnalysis(a) { SparseHouseholderQr.factor(it, symbolic) }
    }

    /** No scaling and no drop tolerance, so `ReferenceSparseDecompositions.factor(a)` reads as the plain routine. */
    public companion object Default : ReferenceSparseDecompositions()
}
