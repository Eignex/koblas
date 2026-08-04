package com.eignex.koblas.dense

import com.eignex.koblas.NOT_SINGULAR
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
class LuDecomposition(val n: Int, val lu: DoubleArray, val piv: IntArray, failedAt: Int = NOT_SINGULAR) {
    /**
     * Where the factorization broke down: the position `k` whose pivot `U[k][k]` was exactly zero, or
     * [NOT_SINGULAR] when it did not. LAPACK reports the same number as `dgetrf`'s positive `info`, one
     * lower for being 0-based, and the reference loop knows it just as directly.
     *
     * Refactorizing in place via [LinearAlgebra.factorInto] updates it along with the buffers.
     */
    var failedAt: Int = failedAt
        internal set

    /**
     * Whether a zero pivot was encountered; [LinearAlgebra.solve] on a singular factorization is not
     * meaningful.
     *
     * Derived from [failedAt] rather than stored, so the two cannot contradict each other. It used to be
     * the stored flag and every backend set it separately, which left the position — the more useful half,
     * and the one a caller can act on — computed and then discarded.
     */
    val singular: Boolean get() = failedAt != NOT_SINGULAR

    init {
        require(lu.size == n * n) { "lu length ${lu.size} != ${n * n}" }
        require(piv.size == n) { "piv length ${piv.size} != $n" }
    }
}

/** `det(A)` from the factorization: `sign(P) · ∏ U[k][k]`, or exactly `0.0` when [LuDecomposition.singular].
 *  The floating-point counterpart of [SparseLu.determinant]. */
fun LuDecomposition.determinant(): Double {
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
