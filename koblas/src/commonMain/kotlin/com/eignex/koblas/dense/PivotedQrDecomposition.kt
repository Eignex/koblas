package com.eignex.koblas.dense

import com.eignex.koblas.requireShape
import kotlin.math.abs

/** The [LinearAlgebra.qrPivoted] tolerance meaning "derive one from the matrix", `max(m, n) · ε`. */
public const val AUTOMATIC_RANK_TOLERANCE: Double = 0.0

/** `2⁻⁵²`, the gap between 1.0 and the next double. Kotlin has no `Double.EPSILON`. */
internal const val MACHINE_EPSILON = 2.220446049250313e-16

/** Below this, a downdated column norm is recomputed from the column instead of trusted. `√ε`, as in
 *  LAPACK `dgeqp3`. */
internal const val NORM_RECOMPUTE_THRESHOLD = 1.4901161193847656e-8

/**
 * A QR factorization with column pivoting, `A·P = Q·R` (LAPACK `dgeqp3`), plus the numerical [rank] the
 * pivoting revealed. [factorization] factorizes `A·P`, so its solutions come out in permuted column order.
 *
 * @property factorization the `Q` and `R` of `A·P`, in the packed `dgeqrf` form.
 * @property pivots pivots(k) is the column of `A` at position k of `A·P`, LAPACK `jpvt` made 0-based.
 * @property rank the number of `R` diagonal entries above the tolerance, in `0..min(m, n)`.
 */
public class PivotedQrDecomposition(
    public val factorization: QrDecomposition,
    public val pivots: IntArray,
    public val rank: Int,
) {
    /** The row count of the factored matrix. */
    public val m: Int get() = factorization.m

    /** The column count of the factored matrix. */
    public val n: Int get() = factorization.n

    /** Whether the numerical rank is below `min(m, n)`. */
    public val rankDeficient: Boolean get() = rank < minOf(m, n)

    init {
        requireShape(pivots.size == n) { "pivots length ${pivots.size} != $n" }
        require(rank in 0..minOf(m, n)) { "rank $rank outside 0..${minOf(m, n)}" }
    }
}

/**
 * The numerical rank of a pivoted `R`, the leading run of diagonal entries above `tolerance · |R₀₀|`.
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
