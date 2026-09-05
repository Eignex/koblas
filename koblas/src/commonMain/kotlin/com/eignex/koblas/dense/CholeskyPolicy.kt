package com.eignex.koblas.dense

/** What [F64Decompositions.cholesky] does when the matrix turns out not to be positive-definite. */
public sealed interface CholeskyPolicy {
    /** Throw at the first non-positive pivot, naming the position and the value. */
    public data object Strict : CholeskyPolicy

    /**
     * Continue past a pivot below [minimumPivot], raising it instead. The result is a genuine Cholesky
     * factor of a nearby matrix, not of the input.
     *
     * The floor applies to any pivot beneath it, not only a non-positive one: a tiny positive pivot is the
     * near-singular case regularizing exists for, and leaving it alone would overflow the column it scales.
     * A raised pivot is lifted further where the column below it demands, so the multipliers it writes stay
     * at or below one and the trailing submatrix is not inflated by `1/minimumPivot`. How many pivots were
     * altered comes back as [com.eignex.koblas.dense.F64CholeskyDecomposition.regularizations]; which ones
     * they were is not currently reported.
     *
     * A NaN pivot is refused under this policy too. It is corrupt input rather than indefiniteness, and no
     * floor repairs it: the column below stays NaN while the diagonal reads clean.
     *
     * @property minimumPivot an absolute floor in the matrix's own units, not the factor's; must be positive.
     */
    public data class Regularize(val minimumPivot: Double = 1e-10) : CholeskyPolicy {
        init {
            require(minimumPivot > 0.0 && minimumPivot.isFinite()) {
                "minimumPivot must be positive and finite, got $minimumPivot"
            }
        }
    }
}
