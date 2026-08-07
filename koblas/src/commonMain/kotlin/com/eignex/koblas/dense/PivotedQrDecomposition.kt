package com.eignex.koblas.dense

/**
 * The [LinearAlgebra.qrPivoted] tolerance meaning "derive one from the matrix": `max(m, n) · ε`.
 *
 * A sentinel rather than a value, because the sensible default depends on the shape — a tolerance is only
 * meaningful against the size of the arithmetic that produced the entry, and a 1000-row matrix accumulates
 * more rounding than a 3-row one.
 */
const val AUTOMATIC_RANK_TOLERANCE = 0.0

/** `2⁻⁵²`, the gap between 1.0 and the next double. Kotlin has no `Double.EPSILON`. */
internal const val MACHINE_EPSILON = 2.220446049250313e-16

/**
 * Below this, a downdated column norm is recomputed from the column instead of trusted.
 *
 * `√ε`, matching LAPACK's `dgeqp3`: the downdate subtracts two nearly equal quantities once a column has
 * lost most of its norm, and past this point the result carries fewer than half its digits.
 */
internal const val NORM_RECOMPUTE_THRESHOLD = 1.4901161193847656e-8

/**
 * A QR factorization with column pivoting, `A·P = Q·R` (LAPACK `dgeqp3`), plus the numerical [rank] the
 * pivoting revealed.
 *
 * A separate type from [QrDecomposition] rather than a nullable pivot array on it, because the two are not
 * interchangeable in the one place it matters: `R` here factorizes `A·P`, so feeding [factorization] to
 * [LinearAlgebra.solveLeastSquares] returns a solution whose entries are in permuted order — a wrong answer
 * that looks entirely plausible. Making it a distinct type means that mistake does not compile; use
 * [LinearAlgebra.solveLeastSquares] with this object instead, which undoes the permutation.
 *
 * [factorization] is the genuine `Q` and `R` of the permuted matrix, so [LinearAlgebra.applyQ] takes it
 * directly. Nothing is copied: the buffers are the ones the factorization produced.
 *
 * Column pivoting is what makes QR able to *report* rank at all. Unpivoted Householder QR is stable but it
 * has no reason to put the dependent columns last, so a rank-deficient matrix leaves small `R` diagonal
 * entries scattered anywhere and there is nothing to threshold. Pivoting always takes the largest remaining
 * column, which drives `|R₀₀| ≥ |R₁₁| ≥ …` and pushes the dependence to the trailing entries, so [rank] is
 * the count of leading diagonal entries above the tolerance.
 *
 * The rank is a *numerical* one and the tolerance is a judgement, not a fact — see
 * [LinearAlgebra.qrPivoted]. A matrix with a genuine gap in its singular values reports the same rank across
 * any sensible tolerance; one without a gap reports whatever the tolerance says, and no factorization can do
 * better. Pivoted QR is also not a rank oracle: it is defeated by contrived matrices (the Kahan matrix being
 * the standard example, where every leading submatrix looks well conditioned), for which an SVD is the
 * honest tool.
 *
 * @property factorization the `Q` and `R` of `A·P`, in the packed `dgeqrf` form.
 * @property pivots `pivots[k]` is the column of `A` that sits at position `k` of `A·P`, LAPACK's `jpvt`
 *  content in 0-based form. A permutation of `0 until n`.
 * @property rank the number of `R` diagonal entries above the tolerance, in `0..min(m, n)`.
 */
class PivotedQrDecomposition(val factorization: QrDecomposition, val pivots: IntArray, val rank: Int) {
    /** The row count of the factored matrix. */
    val m: Int get() = factorization.m

    /** The column count of the factored matrix. */
    val n: Int get() = factorization.n

    /** Whether the pivoting found fewer independent columns than there are columns. */
    val rankDeficient: Boolean get() = rank < minOf(m, n)

    init {
        require(pivots.size == n) { "pivots length ${pivots.size} != $n" }
        require(rank in 0..minOf(m, n)) { "rank $rank outside 0..${minOf(m, n)}" }
    }
}
