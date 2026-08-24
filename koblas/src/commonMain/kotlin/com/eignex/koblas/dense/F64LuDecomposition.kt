package com.eignex.koblas.dense

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.requireShape

/**
 * A general LU factorization with partial pivoting, `P·A = L·U`, packed column-major with `L` below the
 * diagonal and `U` on and above. [lu] and [piv] are live buffers, not copies, so treat them as read-only.
 *
 * @property n the matrix dimension.
 * @property lu the packed `L`\`U` factors, column-major, length `n * n`.
 * @property piv the row permutation, where piv(k) is the original row now at position k.
 * @param failedAt the position of the zero pivot, or [NOT_SINGULAR]; readable afterwards through [failedAt].
 */
public class F64LuDecomposition @UnsafeKoblasApi constructor(
    public val n: Int,
    @property:UnsafeKoblasApi public val lu: DoubleArray,
    @property:UnsafeKoblasApi public val piv: IntArray,
    failedAt: Int = NOT_SINGULAR,
) {
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

    init {
        requireShape(lu.size == n * n) { "lu length ${lu.size} != ${n * n}" }
        requireShape(piv.size == n) { "piv length ${piv.size} != $n" }
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
    var d = permutationSign(piv)
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
