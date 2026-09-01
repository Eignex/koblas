package com.eignex.koblas.dense

import com.eignex.koblas.*

/**
 * A general LU factorization with partial pivoting, `P·A = L·U`, packed column-major with `L` below the
 * diagonal and `U` on and above. [lu] is a live buffer, not a copy, so treat it as read-only.
 *
 * @property n the matrix dimension.
 * @property lu the packed `L`\`U` factors, column-major, length `n * n`.
 * @param piv the live 0-based final row permutation; kept for source compatibility and exposed as the
 * deprecated [piv] property. Prefer [rowPermutation] or [rowAt].
 * @param failedAt the position of the zero pivot, or [NOT_SINGULAR]; readable afterwards through [failedAt].
 */
public class F64LuDecomposition @UnsafeKoblasApi constructor(
    public val n: Int,
    @property:UnsafeKoblasApi public val lu: DoubleArray,
    /**
     * The live, 0-based final row permutation. Prefer [rowPermutation] or [rowAt] so callers cannot mutate
     * the factorization.
     *
     * This property is deprecated. Use [rowPermutation] for a copy or [rowAt] for one position. This live
     * buffer remains only for source compatibility.
     */
    piv: IntArray,
    failedAt: Int = NOT_SINGULAR,
) {
    private val pivotBuffer: IntArray = piv

    /**
     * The live, 0-based final row permutation.
     *
     * This property is deprecated. Use [rowPermutation] for a copy or [rowAt] for one position. This live
     * buffer remains only for source compatibility.
     */
    @Deprecated(
        message = "Use rowPermutation or rowAt; piv is a live compatibility buffer.",
        replaceWith = ReplaceWith("rowPermutation"),
    )
    @UnsafeKoblasApi
    public val piv: IntArray get() = pivotBuffer

    /**
     * The position k whose pivot U(k, k) was exactly zero, or [NOT_SINGULAR]. This is `dgetrf`'s positive
     * `info` made 0-based.
     */
    public var failedAt: Int = failedAt
        internal set

    /**
     * Whether a zero pivot was encountered. [F64LinearAlgebra.solve] then throws
     * [com.eignex.koblas.SingularMatrix] rather than dividing by it.
     */
    public val singular: Boolean get() = failedAt != NOT_SINGULAR

    /** A 0-based copy of the final row permutation: element k is the original row now at factor row k. */
    public val rowPermutation: IntArray get() = pivotBuffer.copyOf()

    /** The original 0-based row now at factor row [position]. */
    public fun rowAt(position: Int): Int {
        requireInBounds(position, n)
        return pivotBuffer[position]
    }

    /** The mutable factorization buffer for internal implementations. */
    internal val mutablePivots: IntArray get() = pivotBuffer

    init {
        requireShape(lu.size == n * n) { "lu length ${lu.size} != ${n * n}" }
        requireShape(pivotBuffer.size == n) { "piv length ${pivotBuffer.size} != $n" }
    }
}

/**
 * `det(A)` as sign(P) times the product of the U(k, k).
 *
 * The product is unscaled, so it saturates in both directions well before n is large: a 200x200 with 0.01
 * on the diagonal returns `0.0` and one with 100.0 returns infinity, neither of them singular. So a
 * returned `0.0` does not mean singular, even though a singular factorization does return it. Test
 * [F64LuDecomposition.singular] for exact singularity and [F64Decompositions.rcond] for how close to it a matrix
 * is; that is what LAPACK offers too, which ships `dgecon` and no determinant routine at all.
 */
public fun F64LuDecomposition.determinant(): Double {
    if (singular) return 0.0
    var d = permutationSign(mutablePivots)
    for (k in 0 until n) d *= lu[k * n + k]
    return d
}

/** Sign of the permutation [p], where p(k) is the original index at position k. */
internal fun permutationSign(p: IntArray): Double {
    val seen = BooleanArray(p.size)
    var cycles = 0
    for (s in p.indices) {
        if (seen[s]) continue
        cycles++
        var i = s
        while (!seen[i]) {
            seen[i] = true
            i = p[i]
        }
    }
    return if ((p.size - cycles) % 2 == 0) 1.0 else -1.0
}
