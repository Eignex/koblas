package com.eignex.koblas.dense

/** What [F64Decompositions.cholesky] does when the matrix turns out not to be positive-definite. */
public sealed interface CholeskyPolicy {
    /** Throw at the first non-positive pivot, naming the position and the value. */
    public data object Strict : CholeskyPolicy

    /**
     * Continue past a non-positive pivot, treating it as [minimumPivot] instead. The result is a genuine
     * Cholesky factor of a nearby matrix, not of the input.
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
