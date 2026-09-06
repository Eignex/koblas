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
 * Why a solver advised [BasisUpdate.REFACTORIZE], for a caller pacing its rebuilds.
 *
 * The advisory is one answer reached two ways, and which one it was is what tells a worn factorization from
 * a numerically unhappy one. A caller logging these can read the difference off a histogram rather than
 * guessing at it.
 */
public enum class F64RefactorizeReason {
    /**
     * The factorization itself asked, having found the update it just took unfit to build on. This is the
     * numerical cause: the factors have lost accuracy rather than merely grown.
     */
    FACTOR_ASKED,

    /**
     * The updates have cost what the factorization did, so rebuilding is now the cheaper path. This is the
     * economic cause and says nothing about accuracy.
     */
    UPDATES_WORN,
}

/**
 * How much of a basis survived triangularization into the elimination kernel.
 *
 * A factorization peels off the rows and columns it can order triangularly before choosing pivots for what
 * is left. What is left is the kernel, and its size against the basis dimension says whether a caller's
 * bases are staying mostly triangular, which a fill count alone does not.
 *
 * @property dimension the kernel's order, at most the basis dimension.
 * @property entries the stored entries in it.
 */
public class F64BasisKernel(public val dimension: Int, public val entries: Int)

/**
 * The basis a solver settled on, where it was allowed to repair one it could not invert as given.
 *
 * A rank-deficient basis has slots whose columns are linearly dependent. A provider that repairs one fills
 * those slots with unit columns, which are not columns of `A` at all: they are the slacks a simplex would
 * hold there. So a repaired basis is reported slot by slot rather than as column indices, since some slots
 * name nothing in the caller's matrix.
 *
 * A caller adopting one takes both arrays as its new basis. Declining it means factorizing something else;
 * the solver holds the repaired basis either way until the next rebuild.
 *
 * @property columns the column of `A` in each slot, or -1 where the slot holds a unit column.
 * @property unitRows the row each unit column stands for, or -1 where the slot holds a column of `A`.
 */
public class F64BasisRepair(public val columns: IntArray, public val unitRows: IntArray) {
    /** Whether anything was replaced. False means the basis factorized exactly as it was given. */
    public val repaired: Boolean get() = unitRows.any { it >= 0 }

    /** How many slots were replaced. */
    public val replacedSlots: Int get() = unitRows.count { it >= 0 }
}

/** Numerically scaled residual information for one basis solve. */
public data class F64BasisSolveQuality(
    /** `max |B·x - b|`, or `max |Bᵀ·x - b|` for a transposed solve. */
    public val residualInfinityNorm: Double,
    /** The residual divided by `max(1, max(|B·x|, |b|))`. */
    public val relativeResidual: Double,
)

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
 * solver serves one simplex. Close it when the simplex is finished; portable implementations own no
 * external resource and use the default no-op [close]. A native solver keeps [n] and [singular] readable
 * after close and rejects factor reports and operations with [IllegalStateException].
 */
public interface F64BasisSolver : AutoCloseable {
    /** The dimension of the basis, the row count of `A`. */
    public val n: Int

    /** Nonzeros in whatever the solver currently holds, counting what the updates have added. */
    public val nnz: Int

    /** Updates folded in since the last [refactorize]. */
    public val updateCount: Int

    /** Whether the last [refactorize] left the solver unable to solve. */
    public val singular: Boolean

    /**
     * A pivot-spread indicator for the current factors. It is zero for a singular basis; a small positive
     * result warns that the basis may need reinversion. This is not a reciprocal condition-number estimate.
     * A native provider may materialize factor metadata to answer it, so sample it rather than polling it.
     */
    public val rcond: Double

    /**
     * Why the last [update] advised [BasisUpdate.REFACTORIZE], or null where it did not or the solver does
     * not say. Reading it after any other outcome answers null.
     */
    public val refactorizeReason: F64RefactorizeReason? get() = null

    /**
     * The elimination kernel the last [refactorize] left, or null where the solver does not report one.
     *
     * A solver whose factorization has no such pass has nothing to say here, which is why this is optional
     * rather than a number every provider has to invent.
     */
    public val kernel: F64BasisKernel? get() = null

    /**
     * Factorizes the basis of [basicIndex], whose length must be [n] and whose entries name columns of `A`,
     * and drops any updates. Returns false when the basis is singular, which leaves the solver unusable
     * until a later call succeeds.
     */
    public fun refactorize(basicIndex: IntArray): Boolean

    /**
     * Factorizes the basis of [basicIndex] as [refactorize] does, but keeps a repair where the provider can
     * make one, rather than reporting the basis singular and leaving the solver unusable.
     *
     * Returns what was factorized, or null where even a repaired basis could not be. A provider that cannot
     * repair answers exactly as [refactorize] did, so a caller may always ask.
     *
     * This exists because refusing a repair costs the caller its warm start: a search whose basis comes back
     * rank deficient after a bound change would otherwise restart from scratch, where adopting a few
     * substituted columns carries on. [refactorize] keeps its own contract for a caller that wants the
     * basis it named or nothing.
     *
     * A repaired basis is generally not in the order it was given, so a caller adopting one reads its slots
     * from the result rather than from the array it passed.
     */
    public fun refactorizeRepairing(basicIndex: IntArray): F64BasisRepair? {
        if (!refactorize(basicIndex)) return null
        return F64BasisRepair(basicIndex.copyOf(), IntArray(basicIndex.size) { -1 })
    }

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
     * Measures the residual of a completed solve against an unmodified copy of its right-hand side.
     * This deliberately does a sparse matrix-vector product, so use it for numerical checks and rebuild
     * decisions rather than on every iteration.
     */
    public fun solveQuality(
        rhs: DoubleArray,
        solution: F64IndexedVector,
        transpose: Boolean = false,
    ): F64BasisSolveQuality

    /**
     * Replace the basis column at slot [pivotRow] with column [entering] of `A`.
     *
     * [spike] is `B⁻¹ A(entering)` under the factorization in effect *before* this call, which a simplex
     * has already computed for its ratio test; passing it back rather than letting the solver recompute it
     * is what keeps an update to one FTRAN. [pivotEta] is `Bᵀ` solved against `e(pivotRow)`, which a dual
     * simplex likewise has in hand as its pivotal row. An implementation that needs [pivotEta] and is not
     * given it computes it, so a primal caller may leave it out rather than manufacture one, and one that
     * takes it takes it on trust: it is not checked against [pivotRow], and an eta computed for some other
     * row folds in a transform that does not invert the basis.
     *
     * Both vectors must arrive as this solver left them. An implementation may recognise its own solve and
     * reuse the form it kept, which is what makes handing them back cheaper than recomputing, so a vector
     * edited in between is read as the value it no longer carries. A caller wanting to hand over something
     * else builds a vector of its own and pays the solve.
     *
     * The caller has already judged the pivot `spike(pivotRow)` acceptable; a solver that disagrees answers
     * [BasisUpdate.SINGULAR] rather than folding in an update it cannot invert. A solver with no basis
     * factorized, or one whose last [refactorize] failed, answers the same: there is nothing to fold into,
     * and the answer says what the caller owes it either way.
     */
    public fun update(
        pivotRow: Int,
        entering: Int,
        spike: F64IndexedVector,
        pivotEta: F64IndexedVector? = null,
    ): BasisUpdate

    /** Releases resources owned by this solver. Portable implementations have nothing to release. */
    override fun close() {}
}
