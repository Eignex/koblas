package com.eignex.koblas.sparse

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.requireShape
import kotlin.math.sqrt

/**
 * What the symmetric factorization does about a pivot it cannot use.
 *
 * The sparse counterpart of `CholeskyPolicy`, with one case that has no dense equivalent: [Indefinite], which
 * is the whole reason to want `L·D·Lᵀ` rather than `L·Lᵀ`. A KKT system is symmetric and not positive
 * definite, and its negative pivots are the answer rather than a failure.
 */
sealed interface SparseLdlPolicy {
    /**
     * Every pivot must be positive, which makes the factorization a Cholesky in scaled form.
     *
     * A non-positive pivot throws, naming the column, exactly as the dense `cholesky` does: a matrix that was
     * supposed to be positive definite and is not is a modelling error, and the useful thing is the column
     * where it stopped being true.
     */
    data object Strict : SparseLdlPolicy

    /** Any nonzero pivot is accepted, negatives included. An exact zero is still singular. */
    data object Indefinite : SparseLdlPolicy

    /**
     * Pivots below [minimumPivot] are raised to it, so a matrix that has drifted slightly indefinite still
     * factors — the caller trading exactness for a factorization that exists. The default matches the dense
     * policy's.
     */
    data class Regularize(val minimumPivot: Double = 1e-10) : SparseLdlPolicy
}

/**
 * A sparse symmetric factorization `A = L·D·Lᵀ`, with `L` unit lower triangular in CSC and `D` a diagonal.
 *
 * One numeric kernel serves both spellings a caller asks for. For a positive-definite matrix this *is* the
 * Cholesky factorization: `L·√D` is the classical `L`, and [SparseLapack.cholesky] is this factorization with
 * [SparseLdlPolicy.Strict] and nothing else. Keeping `D` separate rather than scaling it into `L` costs one
 * multiply per solve and buys the indefinite case, which square roots cannot express.
 *
 * Produced against a [SparseSymbolic] analysis, which is the point of the pair: the pattern is computed once
 * from the graph and reused for as many value updates as the caller has. [symbolic] is exposed so a caller
 * that reached this through [SparseLapack.cholesky] can keep the analysis for the next factorization instead
 * of paying for it again.
 *
 * Unlike [SparseLu] there is no permutation. A symmetric factorization pivots down the diagonal in the order
 * it is given, which is what makes the symbolic phase possible; the price is that a fill-reducing ordering has
 * to be applied by the caller beforehand, and koblas has no AMD to offer yet. On a matrix whose ordering is
 * already reasonable — a banded stiffness matrix, a KKT system assembled in blocks — that costs nothing. On an
 * arbitrary permutation of one it can cost everything, and [SparseSymbolic.nnz] is how a caller finds out
 * before doing the arithmetic.
 */
class SparseLdl internal constructor(
    /** The analysis this factorization was built against, reusable for another matrix of the same pattern. */
    val symbolic: SparseSymbolic,
    private val columnPointers: IntArray,
    private val rowIndices: IntArray,
    private val values: DoubleArray,
    private val diagonal: DoubleArray,
) : SparseFactorization {

    private val permutation: IntArray get() = symbolic.permutation

    override val n: Int get() = diagonal.size

    /** `L`'s off-diagonal entries plus the `n` of `D`, so it counts what the factorization actually stores. */
    override val nnz: Int get() = values.size + diagonal.size

    /** Always false: a [SparseLdl] exists only for a matrix that factored completely. */
    override val failedAt: Int get() = NOT_SINGULAR

    /**
     * Solve `A·x = b` into [out].
     *
     * Three sweeps: `L·y = b` forward, `D·z = y`, then `Lᵀ·x = z` back. [transpose] is accepted and ignored
     * because `A` is symmetric — `Aᵀ·x = b` is the same system, and the flag exists on the interface for the
     * unsymmetric factorizations that need it.
     */
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        // The factors are of P·A·Pᵀ, so the right-hand side is gathered into elimination order and the
        // solution scattered back. A caller never sees the permutation; that is the point of it being the
        // analysis's business rather than theirs.
        val z = if (symbolic.isNatural) {
            if (out !== b) b.copyInto(out)
            out
        } else {
            val staged = workspace?.take(n) ?: DoubleArray(n)
            for (k in 0 until n) staged[k] = b[permutation[k]]
            staged
        }
        // L·y = b, sweeping down each column: once y[j] is final it pushes into the rows below it.
        for (j in 0 until n) {
            val yj = z[j]
            if (yj != 0.0) {
                for (p in columnPointers[j] until columnPointers[j + 1]) z[rowIndices[p]] -= values[p] * yj
            }
        }
        for (j in 0 until n) z[j] /= diagonal[j]
        // Lᵀ·x = z, back up the same columns: column j of L is row j of Lᵀ.
        for (j in n - 1 downTo 0) {
            var s = z[j]
            for (p in columnPointers[j] until columnPointers[j + 1]) s -= values[p] * z[rowIndices[p]]
            z[j] = s
        }
        if (z !== out) {
            for (k in 0 until n) out[permutation[k]] = z[k]
            workspace?.release(z)
        }
        return out
    }

    /**
     * `det(A) = ∏ D`, since `L` is unit triangular and contributes 1.
     *
     * Exact in the sense that no permutation sign is involved — there is no permutation. It can still overflow
     * for a large matrix, which is the honest answer for a value that does not fit a double.
     */
    override fun determinant(): Double {
        var d = 1.0
        for (v in diagonal) d *= v
        return d
    }

    /**
     * The classical Cholesky factor `L·√D`, materialized as a fresh sparse matrix.
     *
     * The factor of `P·A·Pᵀ`, not of `A`, whenever the analysis reordered — `symbolic.permutation` is what
     * relates them, and for [SparseOrdering.Natural] they are the same matrix. Returning the factor of a
     * matrix the caller did not hand in would be worse than saying so.
     *
     * Only defined when every pivot is positive, which [SparseLapack.cholesky] guarantees and
     * [SparseLdlPolicy.Indefinite] does not; a negative pivot throws rather than returning a NaN. Provided
     * because a caller may want the factor itself — to hand to a sampler, or to multiply — rather than a
     * solve, and reconstructing it from `L` and `D` outside this class would mean exposing both.
     */
    fun choleskyFactor(): SparseMatrix {
        val columns = List(n) { j ->
            if (diagonal[j] <= 0.0) {
                throw NotPositiveDefinite(
                    j,
                    diagonal[j],
                    "choleskyFactor needs positive pivots; D[$j] is ${diagonal[j]}. This factorization is " +
                        "L·D·Lᵀ of an indefinite matrix, which has no real Cholesky factor.",
                )
            }
            val scale = sqrt(diagonal[j])
            buildList {
                add(j to scale)
                for (p in columnPointers[j] until columnPointers[j + 1]) add(rowIndices[p] to values[p] * scale)
            }
        }
        return SparseMatrix.ofColumns(n, n, columns)
    }
}

/**
 * The numeric half of the symmetric factorization: up-looking `L·D·Lᵀ`, one row of `L` at a time.
 *
 * Davis's `ldl_numeric`, which is up-looking rather than left-looking because that is what lets it use the
 * pattern the analysis already computed: row `k` of `L` is exactly the elimination-tree path from each `A[i,k]`
 * with `i < k`, walked to the first node this row has already visited. The dense accumulator `y` is scattered
 * with column `k` of `A`, drained down that path, and left at zero for the next row — so it is allocated once
 * for the whole factorization rather than per column.
 *
 * A pivot the [policy] rejects ends the factorization at that column. An exact zero is singular whatever the
 * policy says, and comes back as [SingularSparseFactorization] carrying the column, which is the same contract
 * [SparseLu] reports failure with.
 */
@Suppress("NestedBlockDepth", "ReturnCount") // the up-looking traversal, and one exit per pivot verdict
internal fun numericLdl(
    a: SparseMatrix,
    symbolic: SparseSymbolic,
    policy: SparseLdlPolicy,
): SparseFactorization {
    requireShape(a.rows == a.cols) { "ldl requires a square matrix; got ${a.rows}x${a.cols}" }
    requireShape(a.rows == symbolic.n) {
        "ldl: matrix is ${a.rows}x${a.rows} but the analysis is for ${symbolic.n}"
    }
    // Before anything else: the traversal below follows the analysed pattern's elimination tree, so a matrix
    // with an entry outside that pattern would walk off the end of the tree rather than produce a wrong
    // answer. Checked here, once, instead of defended at every step.
    require(symbolic.describesPatternOf(a)) {
        "ldl: this matrix does not have the pattern the analysis was computed from. Analyse it, or keep the " +
            "structural zeros so the pattern matches."
    }
    // Everything below eliminates in the analysis's order, so a reordered analysis works on the permuted
    // matrix. Natural skips the materialization entirely rather than permuting by the identity.
    val work = if (symbolic.isNatural) a else permutedUpperTriangle(a, symbolic.inversePermutation)
    val n = symbolic.n
    val lp = symbolic.columnPointers
    val parent = symbolic.parent
    val li = IntArray(symbolic.nnz)
    val lx = DoubleArray(symbolic.nnz)
    val filled = IntArray(n) // entries written into each column so far
    val d = DoubleArray(n)
    val y = DoubleArray(n) // dense accumulator, zero between rows by construction
    val path = IntArray(n) // the tree path in topological order, filled from the back
    val climb = IntArray(n) // one ascent, before it is reversed into `path`
    val flag = IntArray(n) { -1 }

    for (k in 0 until n) {
        var top = n
        flag[k] = k
        work.forEachInColumn(k) { row, value ->
            if (row <= k) {
                y[row] += value
                var length = 0
                var node = row
                while (flag[node] != k) {
                    climb[length++] = node
                    flag[node] = k
                    node = parent[node]
                }
                // Reversed into the back of `path`, so the whole row ends up in topological order: a column is
                // drained only after every column that feeds it.
                while (length > 0) path[--top] = climb[--length]
            }
        }
        d[k] = y[k]
        y[k] = 0.0
        for (t in top until n) {
            val i = path[t]
            val yi = y[i]
            y[i] = 0.0
            val end = lp[i] + filled[i]
            for (p in lp[i] until end) y[li[p]] -= lx[p] * yi
            val lki = yi / d[i]
            d[k] -= lki * yi
            check(end < lp[i + 1]) {
                // Not decoration: an analysis that under-counts a column lands here, and writing past the
                // column would corrupt its neighbour instead of failing. Deliberately breaking the count in
                // the symbolic phase trips this, which is how it was verified to be reachable.
                "ldl: column $i needs more room than the analysis reserved it, so the analysis and this " +
                    "matrix disagree about the pattern"
            }
            li[end] = k
            lx[end] = lki
            filled[i]++
        }
        when {
            d[k] == 0.0 -> return SingularSparseFactorization(n, failedAt = k)

            policy is SparseLdlPolicy.Strict && d[k] < 0.0 -> throw NotPositiveDefinite(
                k,
                d[k],
                "ldl: pivot $k is ${d[k]}, so the matrix is not positive definite. Use " +
                    "SparseLdlPolicy.Indefinite to factor it anyway, or Regularize to floor the pivot.",
            )

            policy is SparseLdlPolicy.Regularize && d[k] < policy.minimumPivot -> d[k] = policy.minimumPivot
        }
    }
    return SparseLdl(symbolic, lp, li, lx, d)
}
