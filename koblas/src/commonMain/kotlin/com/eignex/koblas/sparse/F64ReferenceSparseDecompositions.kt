package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.sparse.factorization.cholesky.F64SparseCholeskyFactorization
import com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization
import com.eignex.koblas.sparse.factorization.lu.NO_DROP

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
) : F64SparseDecompositions {
    init {
        require(dropTolerance >= 0.0) { "dropTolerance must not be negative" }
    }

    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    override fun factor(a: F64SparseMatrix): F64SparseFactorization =
        F64SparseLuFactorization.factorCsc(a, equilibrate, dropTolerance)

    /** Neither knob reaches here: equilibration would break the symmetry, and a drop tolerance would leave
     *  an incomplete factor, which is a preconditioner rather than the factorization this promises. */
    override fun cholesky(a: F64SparseMatrix): F64SparseFactorization = F64SparseCholeskyFactorization.factorLower(a)

    /** No scaling and no drop tolerance, so `F64ReferenceSparseDecompositions.factor(a)` reads as the plain routine. */
    public companion object Default : F64ReferenceSparseDecompositions()
}
