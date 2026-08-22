package com.eignex.koblas.sparse.factorization.ldl

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.symbolic.*
import kotlin.math.sqrt

/** What the sparse LDL factorization does with a non-positive pivot. */
public sealed interface SparseLdlPolicy {
    /**
     * Every pivot must be positive, which makes the factorization a Cholesky in scaled form. A non-positive
     * pivot throws, naming the column.
     */
    public data object Strict : SparseLdlPolicy

    /** Any nonzero pivot is accepted, negatives included. An exact zero is still singular. */
    public data object Indefinite : SparseLdlPolicy

    /**
     * Pivots below [minimumPivot], an exact zero included, are raised to it, so this policy always yields a
     * factorization. It is one of a nearby matrix, not of the input.
     *
     * @property minimumPivot an absolute floor in the matrix's own units, not the factor's; must be positive.
     */
    public data class Regularize(val minimumPivot: Double = 1e-10) : SparseLdlPolicy {
        init {
            require(minimumPivot > 0.0 && minimumPivot.isFinite()) {
                "minimumPivot must be positive and finite, got $minimumPivot"
            }
        }
    }
}

/**
 * A sparse symmetric factorization `A = L·D·Lᵀ`, with L unit lower triangular in CSC and D a diagonal.
 * There is no pivoting: elimination runs down the diagonal in the analysis's order.
 */
public class F64SparseLdl internal constructor(
    /** The analysis this factorization was built against, reusable for another matrix of the same pattern. */
    public val symbolic: SparseSymbolic,
    private val columnPointers: IntArray,
    private val rowIndices: IntArray,
    private val values: DoubleArray,
    private val diagonal: DoubleArray,
) : F64SparseFactorization {

    private val permutation: IntArray get() = symbolic.permutation

    override val n: Int get() = diagonal.size

    /** L's off-diagonal entries plus the n of D, so it counts what the factorization actually stores. */
    override val nnz: Int get() = values.size + diagonal.size

    /** Always [NOT_SINGULAR]: a [F64SparseLdl] exists only for a matrix that factored completely. */
    override val failedAt: Int get() = NOT_SINGULAR

    /**
     * Solve `A·x = b` into [out] in three sweeps, `L·y = b` forward, `D·z = y`, then `Lᵀ·x = z` back.
     * [transpose] is accepted and ignored because A is symmetric.
     */
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        // The factors are of P·A·Pᵀ, so b is gathered into elimination order and the solution scattered back.
        val z = if (symbolic.isNatural) {
            if (out !== b) b.copyInto(out)
            out
        } else {
            val staged = workspace?.take(n) ?: DoubleArray(n)
            for (k in 0 until n) staged[k] = b[permutation[k]]
            staged
        }
        // L·y = b, sweeping down each column.
        for (j in 0 until n) {
            val yj = z[j]
            if (yj != 0.0) {
                for (p in columnPointers[j] until columnPointers[j + 1]) z[rowIndices[p]] -= values[p] * yj
            }
        }
        for (j in 0 until n) z[j] /= diagonal[j]
        // Lᵀ·x = z, back up the same columns, since column j of L is row j of Lᵀ.
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

    /** `det(A) = ∏ D`, since L is unit triangular and contributes 1. Overflows for a large matrix. */
    override fun determinant(): Double {
        var d = 1.0
        for (v in diagonal) d *= v
        return d
    }

    /**
     * The classical Cholesky factor `L·√D`, as a fresh sparse matrix. This is the factor of `P·A·Pᵀ`, not of
     * A, whenever the analysis reordered; [SparseSymbolic.copyPermutation] relates them.
     *
     * @throws com.eignex.koblas.NotPositiveDefinite if any pivot is not positive.
     */
    public fun choleskyFactor(): F64SparseMatrix {
        for (j in 0 until n) {
            if (diagonal[j] <= 0.0) {
                throw NotPositiveDefinite(
                    j,
                    diagonal[j],
                    "choleskyFactor needs positive pivots; D[$j] is ${diagonal[j]}. This factorization is " +
                        "L·D·Lᵀ of an indefinite matrix, which has no real Cholesky factor.",
                )
            }
        }
        // The diagonal plus the strict lower triangle, emitted as triplets so no pair is allocated.
        val nnz = n + columnPointers[n]
        val rowIdx = IntArray(nnz)
        val colIdx = IntArray(nnz)
        val scaled = DoubleArray(nnz)
        var k = 0
        for (j in 0 until n) {
            val scale = sqrt(diagonal[j])
            rowIdx[k] = j
            colIdx[k] = j
            scaled[k] = scale
            k++
            for (p in columnPointers[j] until columnPointers[j + 1]) {
                rowIdx[k] = rowIndices[p]
                colIdx[k] = j
                scaled[k] = values[p] * scale
                k++
            }
        }
        return F64SparseMatrix.ofTriplets(n, n, rowIdx, colIdx, scaled)
    }
}

/**
 * The numeric half of the symmetric factorization, up-looking `L·D·Lᵀ` one row of L at a time. An exact zero
 * pivot comes back as [F64SingularSparseFactorization] unless the [policy] floors it.
 */
@Suppress("NestedBlockDepth", "ReturnCount") // the up-looking traversal, and one exit per pivot verdict
internal fun numericLdl(
    a: F64SparseMatrix,
    symbolic: SparseSymbolic,
    policy: SparseLdlPolicy,
): F64SparseFactorization {
    requireSquare(a, "ldl")
    requireShape(a.rows == symbolic.n) {
        "ldl: matrix is ${a.rows}x${a.rows} but the analysis is for ${symbolic.n}"
    }
    // An entry outside the analysed pattern would walk the traversal below off the end of the tree.
    require(symbolic.describesPatternOf(a)) {
        "ldl: this matrix does not have the pattern the analysis was computed from. Analyse it, or keep the " +
            "structural zeros so the pattern matches."
    }
    // Everything below eliminates in the analysis's order, so a reordered analysis needs the permuted matrix.
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
    val climb = IntArray(n) // one ascent, before it is reversed into path
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
                // Reversed into the back of path, so a column is drained only after every column feeding it.
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
                // An under-counted column would be written past, silently corrupting its neighbour.
                "ldl: column $i needs more room than the analysis reserved it, so the analysis and this " +
                    "matrix disagree about the pattern"
            }
            li[end] = k
            lx[end] = lki
            filled[i]++
        }
        when {
            // Ahead of the zero check, since a zero pivot is below any floor.
            policy is SparseLdlPolicy.Regularize && d[k] < policy.minimumPivot -> d[k] = policy.minimumPivot

            // Reported as a column of the caller's matrix. Elimination runs over the permuted one, so k is
            // an ordering-dependent index nothing outside here can act on.
            d[k] == 0.0 -> return F64SingularSparseFactorization(n, failedAt = symbolic.permutation[k])

            policy is SparseLdlPolicy.Strict && d[k] < 0.0 -> throw NotPositiveDefinite(
                symbolic.permutation[k],
                d[k],
                "ldl: the pivot for column ${symbolic.permutation[k]} is ${d[k]}, so the matrix is not " +
                    "positive definite. Use SparseLdlPolicy.Indefinite to factor it anyway, or Regularize " +
                    "to floor the pivot.",
            )
        }
    }
    return F64SparseLdl(symbolic, lp, li, lx, d)
}
