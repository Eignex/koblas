package com.eignex.koblas.dense

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.SparseLu

/**
 * A general LU factorization with partial pivoting: `P·A = L·U`, the unit-lower `L` and upper `U` packed
 * into one flat column-major [lu] buffer (`L` below the diagonal, `U` on and above) and the row permutation
 * in [piv] (`piv[k]` is the original row now at position `k`). Produced by [LinearAlgebra.factor].
 *
 * The constructor and the factor buffers are public so that out-of-repo [LinearAlgebra] backends can
 * produce and consume factorizations in the shared format; every backend packs identically, so a
 * decomposition from one backend solves correctly on another. [lu] and [piv] are live buffers, not
 * copies: treat them as read-only, and note that [LinearAlgebra.factorInto] rewrites them in place so a
 * periodic refactorization need not allocate new ones.
 *
 * @property n the matrix dimension.
 * @property lu the packed `L`\`U` factors, column-major, length `n * n`.
 * @property piv the row permutation.
 * @param failedAt the position of the zero pivot, or [NOT_SINGULAR]; readable afterwards through [failedAt].
 */
public class LuDecomposition(
    public val n: Int,
    public val lu: DoubleArray,
    public val piv: IntArray,
    failedAt: Int = NOT_SINGULAR,
) {
    /**
     * Where the factorization broke down: the position `k` whose pivot `U[k][k]` was exactly zero, or
     * [NOT_SINGULAR] when it did not. LAPACK reports the same number as `dgetrf`'s positive `info`, one
     * lower for being 0-based, and the reference loop knows it just as directly.
     *
     * Refactorizing in place via [LinearAlgebra.factorInto] updates it along with the buffers.
     */
    public var failedAt: Int = failedAt
        internal set

    /**
     * Whether a zero pivot was encountered. [LinearAlgebra.solve] against a singular factorization throws
     * [com.eignex.koblas.SingularMatrix] rather than dividing by the zero pivot, so this is what to check
     * when a singular basis is an expected outcome rather than a bug.
     *
     * Derived from [failedAt] rather than stored, so the two cannot contradict each other and no backend has
     * to remember to set both. The position is the more useful half anyway: it is the one a caller can act
     * on, where a flag only says that something went wrong.
     */
    public val singular: Boolean get() = failedAt != NOT_SINGULAR

    init {
        requireShape(lu.size == n * n) { "lu length ${lu.size} != ${n * n}" }
        requireShape(piv.size == n) { "piv length ${piv.size} != $n" }
    }
}

/** `det(A)` from the factorization: `sign(P) · ∏ U[k][k]`, or exactly `0.0` when [LuDecomposition.singular].
 *  The floating-point counterpart of [SparseLu.determinant]. */
public fun LuDecomposition.determinant(): Double {
    if (singular) return 0.0
    var d = permutationSign(piv)
    for (k in 0 until n) d *= lu[k * n + k]
    return d
}

/** Sign of the permutation [p] (`p[k]` = original index at position `k`): `(-1)^(size − cycles)`. */
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
