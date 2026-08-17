package com.eignex.koblas.sparse

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.permutationSign
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSquare
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.pow

/**
 * Sparse LU factorization `P·B·Q = L·U` of an `m × m` matrix with Markowitz threshold pivoting. The factors
 * are held in both orientations, indexed by pivot position.
 */
public class SparseLu private constructor(
    private val m: Int,
    private val perm: IntArray, // perm(k) is the original row now at pivot position k
    private val colPerm: IntArray, // colPerm(k) is the original column now at pivot position k
    private val lRowIdx: Array<IntArray>, // L by row, pivot positions < k, unit diagonal implicit
    private val lRowVal: Array<DoubleArray>,
    private val uRowIdx: Array<IntArray>, // U by row, pivot positions >= k, the diagonal first in each row
    private val uRowVal: Array<DoubleArray>,
    private val lColIdx: Array<IntArray>, // L by column, pivot positions > k
    private val lColVal: Array<DoubleArray>,
    private val uColIdx: Array<IntArray>, // U by column, pivot positions < k, strictly upper
    private val uColVal: Array<DoubleArray>,
    private val uDiag: DoubleArray,
    /** Per-original-row equilibration factor applied before factorization, so the factors are of `E·B`. All
     *  1.0 when equilibration is off. The solves and [determinant] correct for it. */
    private val rowScale: DoubleArray,
    /** Nonzeros in L and U including the diagonal, the factorization's fill. */
    override val nnz: Int,
) : SparseFactorization {

    override val n: Int get() = m

    /** Always [NOT_SINGULAR]: a [SparseLu] only exists for a matrix that factored completely. */
    override val failedAt: Int get() = NOT_SINGULAR

    /**
     * Solve `B x = b`, or `Bᵀ x = b` when [transpose], into [out]. `b` is indexed by original row and the
     * result by original column.
     */
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        if (transpose) btranInto(b, out, workspace) else ftranInto(b, out, workspace)

    /** The forward direction of [solveInto]: `L U (Qᵀ x) = P (E b)`. */
    private fun ftranInto(b: DoubleArray, out: DoubleArray, workspace: Workspace? = null): DoubleArray {
        requireShape(b.size == m) { "ftran: b size ${b.size} != $m" }
        requireShape(out.size == m) { "ftran: out size ${out.size} != $m" }
        val y = workspace?.take(m) ?: DoubleArray(m)
        val xp = workspace?.take(m) ?: DoubleArray(m)
        // Left uncleared: y reads only below k and xp only above k, so a borrowed buffer's contents are dead.
        // L y = P (E b), in pivot-position space.
        for (k in 0 until m) {
            var s = b[perm[k]] * rowScale[perm[k]]
            val idx = lRowIdx[k]
            val v = lRowVal[k]
            for (t in idx.indices) s -= v[t] * y[idx[t]]
            y[k] = s
        }
        // U x' = y, with x' in pivot-column space.
        for (k in m - 1 downTo 0) {
            var s = y[k]
            val idx = uRowIdx[k]
            val v = uRowVal[k]
            for (t in 1 until idx.size) s -= v[t] * xp[idx[t]] // entry 0 is the diagonal
            xp[k] = s / uDiag[k]
        }
        // x = Q x'. Safe in place, since xp is separate storage from out.
        for (k in 0 until m) out[colPerm[k]] = xp[k]
        workspace?.release(xp)
        workspace?.release(y)
        return out
    }

    /** The transposed sweep of [solveInto]: `Uᵀ Lᵀ (P x) = Q b`. `b` is indexed by original column, the
     *  result by original row. */
    private fun btranInto(b: DoubleArray, out: DoubleArray, workspace: Workspace? = null): DoubleArray {
        requireShape(b.size == m) { "btran: b size ${b.size} != $m" }
        requireShape(out.size == m) { "btran: out size ${out.size} != $m" }
        val z = workspace?.take(m) ?: DoubleArray(m)
        val w = workspace?.take(m) ?: DoubleArray(m)
        // Left uncleared for the reason [ftranInto] gives. z ascends, w descends, each written before read.
        // Uᵀ z = Qᵀ b, forward and lower.
        for (k in 0 until m) {
            var s = b[colPerm[k]]
            val idx = uColIdx[k]
            val v = uColVal[k]
            for (t in idx.indices) s -= v[t] * z[idx[t]]
            z[k] = s / uDiag[k]
        }
        // Lᵀ w = z, back, upper, unit diagonal.
        for (k in m - 1 downTo 0) {
            var s = z[k]
            val idx = lColIdx[k]
            val v = lColVal[k]
            for (t in idx.indices) s -= v[t] * w[idx[t]]
            w[k] = s
        }
        // x = E·x', undoing the row equilibration.
        for (k in 0 until m) out[perm[k]] = w[k] * rowScale[perm[k]]
        workspace?.release(w)
        workspace?.release(z)
        return out
    }

    /**
     * `det(B)` in floating point, as `sign(P)·sign(Q)·∏ uDiag / ∏ e`. The factors are of `E·B`, so the
     * row-equilibration product is divided back out.
     */
    override fun determinant(): Double {
        var d = permutationSign(perm) * permutationSign(colPerm)
        for (k in 0 until m) d *= uDiag[k]
        for (i in 0 until m) d /= rowScale[i] // undo the row equilibration
        return d
    }

    /** Entry points for factorizing. */
    public companion object {

        /**
         * Factorize the square [a], the implementation behind [SparseLapack.factor]. Returns a
         * [SingularSparseFactorization] when no acceptable pivot remains.
         */
        internal fun factorCsc(
            a: SparseMatrix,
            equilibrate: Boolean = false,
            dropTolerance: Double = NO_DROP,
        ): SparseFactorization {
            requireSquare(a, "SparseLu")
            val rows = Array(a.rows) { MutableIntDoubleMap() }
            for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> rows[i].put(j, v) }
            return factorize(rows, a.rows, equilibrate, dropTolerance)
        }

        /**
         * Factorize the `m × m` matrix whose row maps are [rows], where `rows(i)` maps a column to `B(i, j)`.
         * Consumes [rows]: the maps are eliminated in place, so the caller must not reuse them.
         *
         * @param rows the row maps, eliminated in place.
         * @param m the dimension of the square matrix.
         * @param equilibrate scale each row by a power of two near `1/max|row|`, undone in the solves.
         * @param dropTolerance a fraction of the largest magnitude present, applied after equilibration.
         */
        internal fun factorize(
            rows: Array<MutableIntDoubleMap>,
            m: Int,
            equilibrate: Boolean = false,
            dropTolerance: Double = NO_DROP,
        ): SparseFactorization {
            require(dropTolerance >= 0.0) { "dropTolerance must be non-negative; got $dropTolerance" }
            val rowScale = DoubleArray(m) { 1.0 }
            if (equilibrate) {
                for (i in 0 until m) {
                    var maxAbs = 0.0
                    rows[i].forEach { _, value ->
                        val a = abs(value)
                        if (a > maxAbs) maxAbs = a
                    }
                    if (maxAbs <= 0.0) continue
                    val e = 2.0.pow(-floor(log2(maxAbs)).toInt())
                    if (e == 1.0) continue
                    rowScale[i] = e
                    rows[i].scaleValues(e)
                }
            }
            val u = rows // eliminated in place, so u(perm(k)) ends as U's pivot row k
            // Both tolerances are fractions of the largest magnitude present, measured after equilibration.
            val scale = largestMagnitude(u)
            // L multipliers per elimination step, keyed by the eliminated original row.
            val lAtStep = Array(m) { MutableIntDoubleMap() }
            val perm = IntArray(m) { -1 }
            val colPerm = IntArray(m) { -1 }
            val state = MarkowitzState(u, m, negligible = NEGLIGIBLE * scale, dropBelow = dropTolerance * scale)
            for (k in 0 until m) {
                if (!state.selectPivot()) return SingularSparseFactorization(m, failedAt = k)
                perm[k] = state.pivotRow
                colPerm[k] = state.pivotCol
                state.eliminate(lAtStep[k])
            }
            return freeze(u, lAtStep, perm, colPerm, m, rowScale)
        }

        /** The keys of [map] passed through [transform], sorted ascending. */
        private inline fun sortedKeysOf(map: MutableIntDoubleMap, transform: (Int) -> Int): IntArray {
            val out = IntArray(map.size)
            var t = 0
            map.forEach { key, _ -> out[t++] = transform(key) }
            out.sort()
            return out
        }

        @Suppress("LongParameterList")
        private fun freeze(
            u: Array<MutableIntDoubleMap>,
            lAtStep: Array<MutableIntDoubleMap>,
            perm: IntArray,
            colPerm: IntArray,
            m: Int,
            rowScale: DoubleArray,
        ): SparseLu {
            val invPerm = inverseOf(perm)
            val invColPerm = inverseOf(colPerm)
            val uDiag = DoubleArray(m) { k -> u[perm[k]].getOrDefault(colPerm[k], 0.0) }
            val (uRowIdx, uRowVal) = uRowsOf(u, perm, colPerm, invColPerm, m)
            val (lRowIdx, lRowVal) = lRowsOf(lAtStep, invPerm, m)
            // Column orientations in pivot space, U strictly upper and L entire.
            val uCol = columnOrientation(m, uRowIdx, uRowVal, strictlyAbovePivot = true)
            val lCol = columnOrientation(m, lRowIdx, lRowVal, strictlyAbovePivot = false)
            var nnz = 0
            for (k in 0 until m) nnz += uRowIdx[k].size + lRowIdx[k].size
            return SparseLu(
                m, perm, colPerm, lRowIdx, lRowVal, uRowIdx, uRowVal,
                lCol.indices, lCol.values, uCol.indices, uCol.values,
                uDiag, rowScale, nnz,
            )
        }

        /**
         * U's rows in pivot space, each row's columns mapped to pivot positions (all at least k), with the
         * diagonal first and the rest ascending.
         */
        private fun uRowsOf(
            u: Array<MutableIntDoubleMap>,
            perm: IntArray,
            colPerm: IntArray,
            invColPerm: IntArray,
            m: Int,
        ): Pair<Array<IntArray>, Array<DoubleArray>> {
            val idx = Array(m) { k -> sortedKeysOf(u[perm[k]]) { invColPerm[it] } }
            val values = Array(m) { k ->
                val row = u[perm[k]]
                DoubleArray(idx[k].size) { t -> row.getOrDefault(colPerm[idx[k][t]], 0.0) }
            }
            return idx to values
        }

        /** L's rows in pivot space: row k holds the multipliers from each step j < k that eliminated it. */
        private fun lRowsOf(
            lAtStep: Array<MutableIntDoubleMap>,
            invPerm: IntArray,
            m: Int,
        ): Pair<Array<IntArray>, Array<DoubleArray>> {
            val rowMap = Array(m) { MutableIntDoubleMap() }
            for (j in 0 until m) {
                lAtStep[j].forEach { origRow, f -> rowMap[invPerm[origRow]].put(j, f) }
            }
            val idx = Array(m) { k -> sortedKeysOf(rowMap[k]) { it } }
            val values = Array(m) { k ->
                DoubleArray(idx[k].size) { t -> rowMap[k].getOrDefault(idx[k][t], 0.0) }
            }
            return idx to values
        }

        /**
         * Transpose a row-indexed factor into column arrays. Walking k ascending leaves each column's rows
         * ascending, which both solve directions assume.
         *
         * @param m the dimension of the square matrix.
         * @param rowIdx the column indices stored in each row.
         * @param rowVal the values parallel to [rowIdx].
         * @param strictlyAbovePivot selects U's strict upper triangle; L keeps every entry.
         */
        private fun columnOrientation(
            m: Int,
            rowIdx: Array<IntArray>,
            rowVal: Array<DoubleArray>,
            strictlyAbovePivot: Boolean,
        ): ColumnOrientation {
            val counts = IntArray(m)
            for (k in 0 until m) {
                val idx = rowIdx[k]
                for (t in idx.indices) {
                    val col = idx[t]
                    if (!strictlyAbovePivot || col > k) counts[col]++
                }
            }
            val outIdx = Array(m) { IntArray(counts[it]) }
            val outVal = Array(m) { DoubleArray(counts[it]) }
            val next = IntArray(m)
            for (k in 0 until m) {
                val idx = rowIdx[k]
                val v = rowVal[k]
                for (t in idx.indices) {
                    val col = idx[t]
                    if (!strictlyAbovePivot || col > k) {
                        val slot = next[col]++
                        outIdx[col][slot] = k
                        outVal[col][slot] = v[t]
                    }
                }
            }
            return ColumnOrientation(outIdx, outVal)
        }
    }
}
