package com.eignex.koblas.dense

/**
 * What [Lapack.cholesky] does when the matrix turns out not to be positive-definite.
 *
 * A named policy rather than a boolean flag, and [Strict] rather than [Regularize] by default. Both parts
 * are deliberate corrections.
 *
 * koblas used to regularize *by default*: a non-positive pivot was quietly replaced and factorization
 * continued, so `a.cholesky()` returned a factor of a matrix that was not the one you passed. That is
 * genuinely useful for the workload it was written for — online statistics on a drifting precision matrix,
 * where the alternative is an exception every few updates — but it is a surprising default for a general
 * linear algebra library, and it is not what `dpotrf` does. LAPACK reports the failing leading minor.
 * Silently returning something plausible is the worse failure: the caller gets numbers, not a signal.
 *
 * The type also gives the fudge factor a name. It used to be a bare `1e-5` in the middle of the
 * elimination loop, with no way to see it, change it, or find out that it existed.
 */
sealed interface CholeskyPolicy {
    /**
     * Throw at the first non-positive pivot, naming the position and the value. The default, and what
     * `dpotrf` effectively does.
     */
    data object Strict : CholeskyPolicy

    /**
     * Continue past a non-positive pivot, treating it as [minimumPivot] instead.
     *
     * The result is a genuine Cholesky factor of a *nearby* matrix, not of the input — which is the point
     * for an iterative estimate that has drifted slightly indefinite, and is why the caller has to ask.
     *
     * [minimumPivot] is in the matrix's own units, not the factor's: the diagonal of `L` becomes its square
     * root. The default reproduces koblas's historical behaviour exactly (`1e-10` here was the buried
     * `1e-5` on `L`). It is an absolute floor rather than one scaled to the matrix, so a badly scaled input
     * wants a value chosen for it — the reason it is a parameter at all.
     *
     * @property minimumPivot the value a non-positive pivot is raised to; must be positive.
     */
    data class Regularize(val minimumPivot: Double = 1e-10) : CholeskyPolicy {
        init {
            require(minimumPivot > 0.0 && minimumPivot.isFinite()) {
                "minimumPivot must be positive and finite, got $minimumPivot"
            }
        }
    }
}
