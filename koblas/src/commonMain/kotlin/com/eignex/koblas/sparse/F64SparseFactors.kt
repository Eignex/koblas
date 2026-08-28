package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseMatrix

/*
 * The factors each kind of sparse factorization produces, on the interface rather than on whichever concrete
 * class happens to implement it. A caller reaching a backend through the seam can read them whoever answered.
 *
 * One interface per kind rather than one shared pair of accessors, because the kinds differ in what they
 * produce: a Cholesky has no separate `U`, an `L·D·Lᵀ` has a `D` instead, and a QR is `m×n` and holds `Q` as
 * an operator. A single `l`/`u` pair would be null half the time.
 *
 * Every accessor here is materialised on first read. A native binding pays a copy for it, and SPQR pays more
 * than that, so a caller who only solves never pays at all. Reads go through the same lifecycle as a solve,
 * so a closed factorization refuses them, and a singular factorization has no factors to give.
 *
 * A provider whose factors are not a matrix it can hand back raises [FactorsNotExposed] rather than
 * inventing one. BASICLU and HFactor do: they keep a basis representation for updating rather than an `L` and
 * a `U`, which is what they are for.
 */

/** Raised by a factorization whose provider keeps its factors in a form it cannot hand back. */
public class FactorsNotExposed(factor: String) :
    UnsupportedOperationException("this factorization does not expose $factor")

/**
 * The factors a general sparse LU produces, indexed by pivot position.
 *
 * The identity they satisfy is `L·U + F = P·diag(rowScaling)·A·Q`, where `P` is [rowOrder], `Q` is
 * [columnOrder] and `F` is [offDiagonal]. [rowScaling] multiplies rather than divides, whatever a library's
 * own convention, so the identity reads one way everywhere.
 *
 * `F` is empty for a factorization that eliminates over the whole matrix at once, which is most of them, and
 * the identity then reads as the `L·U` one would expect. It is there for a provider that permutes to block
 * triangular form first and factors the blocks: KLU does, and its off-diagonal blocks are neither in `L` nor
 * in `U`.
 */
public interface F64SparseLuFactorization : F64SparseFactorization {
    /** Unit lower triangular, its diagonal stored. */
    public val l: F64SparseMatrix get() = throw FactorsNotExposed("l")

    /** Upper triangular, its diagonal stored. */
    public val u: F64SparseMatrix get() = throw FactorsNotExposed("u")

    /** The original row now at each pivot position, the `P` above. */
    public val rowOrder: IntArray get() = throw FactorsNotExposed("rowOrder")

    /** The original column now at each pivot position, the `Q` above. */
    public val columnOrder: IntArray get() = throw FactorsNotExposed("columnOrder")

    /**
     * The per-original-row factor the matrix was multiplied by before factorizing. All ones where the backend
     * does not equilibrate, so the identity above needs no special case for it.
     */
    public val rowScaling: DoubleArray get() = throw FactorsNotExposed("rowScaling")

    /**
     * The entries outside the diagonal blocks, `F` in the identity above. Empty unless the provider factored
     * a block triangular form, which is the unusual case.
     */
    public val offDiagonal: F64SparseMatrix
        get() = F64SparseMatrix.wrap(n, n, IntArray(n + 1), IntArray(0), DoubleArray(0))
}

/** The `A = L·Lᵀ` a sparse Cholesky produces. */
public interface F64SparseCholeskyFactorization : F64SparseFactorization {
    /** Lower triangular, its diagonal stored. */
    public val l: F64SparseMatrix

    /** The original row and column at each position of [l], the same permutation for both by symmetry. */
    public val order: IntArray
}

/** The `A = L·D·Lᵀ` a sparse `L·D·Lᵀ` produces. */
public interface F64SparseLdlFactorization : F64SparseFactorization {
    /**
     * Unit lower triangular, its diagonal of ones implicit rather than stored: [d] occupies that position in
     * every library that keeps one, so storing ones there would be `n` entries carrying no information. A
     * caller reconstructing `A` supplies them, which is what the conformance helpers do.
     *
     * Unlike the `L` of an [F64SparseLuFactorization], whose diagonal is stored because the libraries hand it
     * back that way.
     */
    public val l: F64SparseMatrix

    /** The diagonal factor, one entry per column of [l]. */
    public val d: DoubleArray

    /** Counts of positive, negative, and zero entries in [d]. */
    public val inertia: FactorizationInertia

    /** The original row and column at each position of [l], the same permutation for both by symmetry. */
    public val order: IntArray
}
