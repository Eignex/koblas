package com.eignex.koblas.sparse.basis

/** What a basis solver made of an update, and what the caller owes it before the next solve. */
public enum class BasisUpdate {
    /** The factors now represent the new basis and are fit to solve against. */
    APPLIED,

    /**
     * The factors represent the new basis, but the solver would rather be rebuilt than take another
     * update. Advisory: solving against them stays correct, and the caller paces its own refactorizations.
     */
    REFACTORIZE,

    /**
     * The pivot was not usable and the factors are unchanged, so the basis and the solver have parted
     * company. Only [F64BasisSolver.refactorize] recovers.
     */
    SINGULAR,
}

/**
 * The factorization of a simplex basis, held across pivots.
 *
 * A basis is a choice of [n] columns of a matrix `A` fixed for the solver's lifetime, named by index:
 * [refactorize] takes the whole choice, [update] moves one column of it. A caller therefore never assembles
 * a square basis matrix per refactorization, and a backend that factors `A`'s columns where they lie never
 * has one to read. `A` carries the logical columns explicitly, so a basis slot naming a logical is an
 * ordinary column rather than a case for the solver to know about.
 *
 * Solves go through [F64IndexedVector] rather than `DoubleArray` so a result that is sparse stays sparse
 * across the seam. On a large model a simplex iteration touches a few positions, and a dense carrier would
 * spend `O(n)` clearing and rescanning around work that is `O(1)` in the model's size.
 *
 * The division of labour: this owns the factors, the pivot order, and the updates, and reports when they
 * are wearing out. A simplex owns pricing, the ratio tests, and the refactorization policy that reads those
 * reports. Nothing here sees a cost, a bound, or an objective.
 *
 * A solver is mutable and single-threaded: [update] and [refactorize] change the factors in place, so one
 * solver serves one simplex.
 */
public interface F64BasisSolver {
    /** The dimension of the basis, the row count of `A`. */
    public val n: Int

    /** Nonzeros in whatever the solver currently holds, counting what the updates have added. */
    public val nnz: Int

    /** Updates folded in since the last [refactorize]. */
    public val updateCount: Int

    /** Whether the last [refactorize] left the solver unable to solve. */
    public val singular: Boolean

    /**
     * Factorizes the basis of [basicIndex], whose length must be [n] and whose entries name columns of `A`,
     * and drops any updates. Returns false when the basis is singular, which leaves the solver unusable
     * until a later call succeeds.
     */
    public fun refactorize(basicIndex: IntArray): Boolean

    /**
     * Solve `B x = b` in place: [x] carries `b` in and `x` out.
     *
     * [expectedDensity] is the caller's guess at the density of the result, from what its previous
     * iterations produced. It steers the choice of sweep and nothing else, so a caller with no estimate
     * passes the default and gets a correct answer by the route that suits a dense one.
     */
    public fun ftran(x: F64IndexedVector, expectedDensity: Double = 1.0)

    /** Solve `Bᵀ x = b` in place, the transposed counterpart of [ftran]. */
    public fun btran(x: F64IndexedVector, expectedDensity: Double = 1.0)

    /**
     * Replace the basis column at slot [pivotRow] with column [entering] of `A`.
     *
     * [spike] is `B⁻¹ A(entering)` under the factorization in effect *before* this call, which a simplex
     * has already computed for its ratio test; passing it back rather than letting the solver recompute it
     * is what keeps an update to one FTRAN. [pivotEta] is `Bᵀ` solved against `e(pivotRow)`, which a dual
     * simplex likewise has in hand as its pivotal row. An implementation that needs [pivotEta] and is not
     * given it computes it, so a primal caller may leave it out rather than manufacture one.
     *
     * The caller has already judged the pivot `spike(pivotRow)` acceptable; a solver that disagrees answers
     * [BasisUpdate.SINGULAR] rather than folding in an update it cannot invert.
     */
    public fun update(
        pivotRow: Int,
        entering: Int,
        spike: F64IndexedVector,
        pivotEta: F64IndexedVector? = null,
    ): BasisUpdate
}
