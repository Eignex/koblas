package com.eignex.koblas.dense

import kotlin.math.abs

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
 * A distinct type rather than a nullable pivot array on [QrDecomposition], because `R` here factorizes `A·P`:
 * passing [factorization] to the unpivoted [LinearAlgebra.solveLeastSquares] would return a solution in
 * permuted order, a plausible-looking wrong answer. As a separate type that mistake does not compile. The `Q`
 * and `R` inside are genuine, so [LinearAlgebra.applyQ] takes them directly and nothing is copied.
 *
 * Pivoting is what lets QR report rank at all: always taking the largest remaining column drives
 * `|R₀₀| ≥ |R₁₁| ≥ …`, so the dependent columns collect at the end and [rank] is the leading run above the
 * tolerance. Unpivoted QR scatters small diagonal entries anywhere and gives nothing to threshold.
 *
 * The rank is numerical and the tolerance a judgement: a matrix with a real gap in its singular values reports
 * the same rank across any sensible tolerance, one without reports whatever the tolerance says. Contrived
 * matrices defeat it outright — the Kahan matrix is the standard example — and want an SVD instead.
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

/**
 * The numerical rank of a pivoted `R`: the leading run of diagonal entries above `tolerance · |R₀₀|`.
 *
 * Shared by every backend, since `dgeqp3` reports no rank and two implementations of one tolerance rule would
 * eventually disagree. A leading run rather than a count above the bound: the pivoted diagonal is
 * non-increasing, so an entry above the threshold after one below it would mean the pivoting had gone wrong.
 * [r] is the packed buffer, column-major with `lda == m`, and `k = min(m, n)`.
 */
internal fun rankOfPivotedR(r: DoubleArray, m: Int, n: Int, k: Int, tolerance: Double): Int {
    if (k == 0) return 0
    val largest = abs(r[0])
    if (largest == 0.0) return 0
    val effective = if (tolerance > AUTOMATIC_RANK_TOLERANCE) tolerance else maxOf(m, n) * MACHINE_EPSILON
    val limit = effective * largest
    var rank = 0
    while (rank < k && abs(r[rank + rank * m]) > limit) rank++
    return rank
}
