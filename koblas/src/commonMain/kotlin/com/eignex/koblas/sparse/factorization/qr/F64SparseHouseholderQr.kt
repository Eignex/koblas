package com.eignex.koblas.sparse.factorization.qr

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.AllocationGuarantee
import com.eignex.koblas.ScratchKind
import com.eignex.koblas.ScratchRequirement
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.F64_MACHINE_EPSILON
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSolveShapes
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.internal.transposeCsc
import com.eignex.koblas.sparse.internal.transposeRaw
import kotlin.math.abs
import kotlin.math.hypot

/**
 * Sparse QR by Householder reflections, the portable definition the bindings are measured against.
 *
 * `Q` is held as the reflections rather than formed. Rows are permuted so column `k` pivots at row `k`, and a
 * column with no row left takes a fictitious one, which is why [rows] can exceed [m]. Columns are not
 * reordered, for the reason the portable Cholesky gives about its own ordering.
 */
public class F64SparseHouseholderQr internal constructor(
    override val m: Int,
    override val n: Int,
    private val rows: Int,
    private val rowPermutation: IntArray,
    private val vColPtr: IntArray,
    private val vRowIdx: IntArray,
    private val vValues: DoubleArray,
    private val beta: DoubleArray,
    private val rColPtr: IntArray,
    private val rRowIdx: IntArray,
    private val rValues: DoubleArray,
) : F64SparseQrFactorization {

    override val rank: Int = numericalRank(m, n, rColPtr, rValues)

    override val nnz: Int get() = vValues.size + rValues.size

    override val columnOrder: IntArray get() = IntArray(n) { it }

    // R comes out of the numeric pass in elimination-path order, so its columns need sorting. A double
    // transpose is what sorts a CSC matrix, and that is exactly what transposeRaw does, twice.
    override val r: F64SparseMatrix
        get() = transposeCsc(transposeRaw(n, n, rColPtr, rRowIdx, rValues))

    /**
     * Where row `i` of `A` sits, for the first [m] rows.
     *
     * These index the factorization's own row space, which the symbolic analysis widens past [m] whenever a
     * column has no row left to pivot on, so an entry may be [m] or larger and must not be used to index a
     * caller's length-[m] array.
     */
    public val rowOrder: IntArray get() = rowPermutation.copyOf(m)

    /** Built once, as the Markowitz LU does: a caller reaches the strict solve to avoid allocating. */
    private val solveAllocation = AllocationCapability(
        AllocationGuarantee.NO_MANAGED,
        listOf(ScratchRequirement(ScratchKind.F64, rows)),
    )

    override fun solveAllocation(): AllocationCapability = solveAllocation

    override fun solveInto(b: DoubleArray, out: DoubleArray, workspace: Workspace?): DoubleArray {
        requireSolveShapes(m, n, b, out)
        if (rankDeficient) {
            throw SingularMatrix(rank, "solve: the QR factorization has rank $rank of $n columns")
        }
        workspace.borrow(rows) { x ->
            x.fill(0.0, 0, rows)
            for (i in 0 until m) x[rowPermutation[i]] = b[i]
            for (k in 0 until n) applyReflection(k, x)
            for (k in n - 1 downTo 0) {
                val last = rColPtr[k + 1] - 1
                val xk = x[k] / rValues[last]
                x[k] = xk
                if (xk != 0.0) {
                    for (p in rColPtr[k] until last) x[rRowIdx[p]] -= rValues[p] * xk
                }
            }
            x.copyInto(out, 0, 0, n)
        }
        return out
    }

    override fun applyQInto(y: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireShape(y.size == m) { "applyQ: y size ${y.size}, expected $m" }
        requireShape(out.size == m) { "applyQ: out size ${out.size}, expected $m" }
        // The symbolic analysis appends a pivot row for every column that has none, so the reflections act
        // on `rows` dimensions rather than m, and Q stops being an operator on R^m: real rows then sit at
        // permuted positions at or past m, where truncating to m would drop them and read fictitious ones
        // in their place. Refused rather than approximated, the way solveInto refuses a rank-deficient
        // solve. solveInto itself stays valid because it only ever reads back the leading n entries.
        if (rows != m) {
            throw SingularMatrix(
                rank,
                "applyQ: the factorization spans $rows rows for a matrix of $m, so Q is not an operator " +
                    "on $m entries; its rank is $rank of $n columns",
            )
        }
        workspace.borrow(rows) { x ->
            x.fill(0.0, 0, rows)
            if (transpose) {
                for (i in 0 until m) x[rowPermutation[i]] = y[i]
                for (k in 0 until n) applyReflection(k, x)
                x.copyInto(out, endIndex = m)
            } else {
                y.copyInto(x, endIndex = m)
                for (k in n - 1 downTo 0) applyReflection(k, x)
                for (i in 0 until m) out[i] = x[rowPermutation[i]]
            }
        }
        return out
    }

    private fun applyReflection(k: Int, x: DoubleArray) {
        var tau = 0.0
        for (p in vColPtr[k] until vColPtr[k + 1]) tau += vValues[p] * x[vRowIdx[p]]
        tau *= beta[k]
        if (tau == 0.0) return
        for (p in vColPtr[k] until vColPtr[k + 1]) x[vRowIdx[p]] -= vValues[p] * tau
    }

    /** Factories. */
    public companion object {
        /** Factor [a], which must have at least as many rows as columns. */
        public fun factor(a: F64SparseMatrix): F64SparseHouseholderQr = factor(a, analyze(a))

        /** The pattern-only half of [factor], for a caller that will factor this structure again. */
        internal fun analyze(a: F64SparseMatrix): SparseQrSymbolic {
            requireShape(a.rows >= a.cols) {
                "qr: A is ${a.rows}x${a.cols}, which is wider than it is tall; factor its transpose instead"
            }
            return analyzeQr(a)
        }

        /** [factor] against an analysis of the same pattern, which the caller has already checked. */
        internal fun factor(a: F64SparseMatrix, symbolic: SparseQrSymbolic): F64SparseHouseholderQr {
            val upperNonzeros = symbolic.upperNonzeros
            val n = a.cols
            val vColPtr = IntArray(n + 1)
            val vRowIdx = IntArray(symbolic.householderNonzeros)
            val vValues = DoubleArray(symbolic.householderNonzeros)
            val rColPtr = IntArray(n + 1)
            val rRowIdx = IntArray(upperNonzeros)
            val rValues = DoubleArray(upperNonzeros)
            val beta = DoubleArray(n)
            factorNumeric(a, symbolic, vColPtr, vRowIdx, vValues, beta, rColPtr, rRowIdx, rValues)
            return F64SparseHouseholderQr(
                m = a.rows,
                n = n,
                rows = symbolic.rows,
                rowPermutation = symbolic.rowPermutation,
                vColPtr = vColPtr,
                vRowIdx = vRowIdx,
                vValues = vValues,
                beta = beta,
                rColPtr = rColPtr,
                rRowIdx = rRowIdx,
                rValues = rValues,
            )
        }
    }
}

private fun numericalRank(m: Int, n: Int, colPtr: IntArray, values: DoubleArray): Int {
    var maximum = 0.0
    for (k in 0 until n) maximum = maxOf(maximum, abs(values[colPtr[k + 1] - 1]))
    val tolerance = maxOf(m, n) * F64_MACHINE_EPSILON * maximum
    return (0 until n).count { abs(values[colPtr[it + 1] - 1]) > tolerance }
}

/**
 * The numeric factorization. `V(:, k)` picks up the rows `A` puts below the diagonal plus those inherited
 * from the children of `k`, since a reflection applied at a child leaves entries a later one has to carry.
 */
@Suppress("LongParameterList", "CyclomaticComplexMethod") // the two factors being filled entry by entry
private fun factorNumeric(
    a: F64SparseMatrix,
    symbolic: SparseQrSymbolic,
    vColPtr: IntArray,
    vRowIdx: IntArray,
    vValues: DoubleArray,
    beta: DoubleArray,
    rColPtr: IntArray,
    rRowIdx: IntArray,
    rValues: DoubleArray,
) {
    val n = a.cols
    val parent = symbolic.parent
    val leftmost = symbolic.leftmost
    val permutation = symbolic.rowPermutation
    // Columns and the rows they pivot on share the low indices, so one mark array stamps both.
    val mark = IntArray(symbolic.rows) { -1 }
    val path = IntArray(n)
    val x = DoubleArray(symbolic.rows)
    var rnz = 0
    var vnz = 0
    for (k in 0 until n) {
        rColPtr[k] = rnz
        val start = vnz
        vColPtr[k] = start
        mark[k] = k
        vRowIdx[vnz++] = k
        var top = n
        a.forEachInColumn(k) { row, value ->
            var i = leftmost[row]
            var length = 0
            while (i != -1 && mark[i] != k) {
                path[length++] = i
                mark[i] = k
                i = parent[i]
            }
            while (length > 0) path[--top] = path[--length]
            val permuted = permutation[row]
            x[permuted] = value
            if (permuted > k && mark[permuted] < k) {
                vRowIdx[vnz++] = permuted
                mark[permuted] = k
            }
        }
        for (p in top until n) {
            val i = path[p]
            applyReflectionTo(vColPtr, vRowIdx, vValues, i, beta[i], x)
            rRowIdx[rnz] = i
            rValues[rnz++] = x[i]
            x[i] = 0.0
            if (parent[i] == k) {
                for (q in vColPtr[i] until vColPtr[i + 1]) {
                    val row = vRowIdx[q]
                    if (mark[row] < k) {
                        mark[row] = k
                        vRowIdx[vnz++] = row
                    }
                }
            }
        }
        for (p in start until vnz) {
            vValues[p] = x[vRowIdx[p]]
            x[vRowIdx[p]] = 0.0
        }
        rRowIdx[rnz] = k
        rValues[rnz++] = householder(vValues, start, vnz - start, beta, k)
    }
    rColPtr[n] = rnz
    vColPtr[n] = vnz
}

private fun applyReflectionTo(
    vColPtr: IntArray,
    vRowIdx: IntArray,
    vValues: DoubleArray,
    k: Int,
    beta: Double,
    x: DoubleArray,
) {
    var tau = 0.0
    for (p in vColPtr[k] until vColPtr[k + 1]) tau += vValues[p] * x[vRowIdx[p]]
    tau *= beta
    if (tau == 0.0) return
    for (p in vColPtr[k] until vColPtr[k + 1]) x[vRowIdx[p]] -= vValues[p] * tau
}

private fun householder(v: DoubleArray, offset: Int, length: Int, beta: DoubleArray, k: Int): Double {
    if (length == 0) {
        beta[k] = 0.0
        return 0.0
    }
    val tailNorm = euclideanNorm(v, offset + 1, length - 1)
    val head = v[offset]
    if (tailNorm == 0.0) {
        beta[k] = 0.0
        v[offset] = 1.0
        return head
    }
    val norm = hypot(head, tailNorm)
    val diagonal = if (head >= 0.0) -norm else norm
    val leading = head - diagonal
    v[offset] = 1.0
    for (i in offset + 1 until offset + length) v[i] /= leading
    beta[k] = (diagonal - head) / diagonal
    return diagonal
}
