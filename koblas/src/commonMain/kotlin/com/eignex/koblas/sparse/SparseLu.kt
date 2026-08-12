package com.eignex.koblas.sparse

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.determinant
import com.eignex.koblas.dense.permutationSign
import com.eignex.koblas.koblas
import com.eignex.koblas.requireShape
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
        requireShape(b.size == m) { "solve: b size ${b.size} != $m" }
        requireShape(out.size == m) { "solve: out size ${out.size} != $m" }
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
            requireShape(a.rows == a.cols) { "SparseLu requires a square matrix; got ${a.rows}x${a.cols}" }
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

        @Suppress("LongMethod", "LongParameterList")
        private fun freeze(
            u: Array<MutableIntDoubleMap>,
            lAtStep: Array<MutableIntDoubleMap>,
            perm: IntArray,
            colPerm: IntArray,
            m: Int,
            rowScale: DoubleArray,
        ): SparseLu {
            val invPerm = IntArray(m).also { for (k in 0 until m) it[perm[k]] = k }
            val invColPerm = IntArray(m).also { for (k in 0 until m) it[colPerm[k]] = k }
            val uDiag = DoubleArray(m) { k -> u[perm[k]].getOrDefault(colPerm[k], 0.0) }
            // U row k in pivot space, its columns mapped to pivot positions (all at least k), with the
            // diagonal first and the rest ascending.
            val uRowIdx = Array(m) { k -> sortedKeysOf(u[perm[k]]) { invColPerm[it] } }
            val uRowVal = Array(m) { k ->
                val row = u[perm[k]]
                DoubleArray(uRowIdx[k].size) { t -> row.getOrDefault(colPerm[uRowIdx[k][t]], 0.0) }
            }
            // L row k in pivot space holds the multipliers from each step j < k that eliminated row perm(k).
            val lRowMap = Array(m) { MutableIntDoubleMap() }
            for (j in 0 until m) {
                lAtStep[j].forEach { origRow, f -> lRowMap[invPerm[origRow]].put(j, f) }
            }
            val lRowIdx = Array(m) { k -> sortedKeysOf(lRowMap[k]) { it } }
            val lRowVal = Array(m) { k ->
                DoubleArray(lRowIdx[k].size) { t -> lRowMap[k].getOrDefault(lRowIdx[k][t], 0.0) }
            }
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

/** One factor's entries indexed by column, parallel per column. */
private class ColumnOrientation(val indices: Array<IntArray>, val values: Array<DoubleArray>)

/**
 * A magnitude this far below the matrix's largest is treated as zero rather than as a pivot candidate. This
 * is a zero test, relative rather than absolute; stability is [PIVOT_THRESHOLD]'s job.
 */
private const val NEGLIGIBLE = 1e-14

/** The default [SparseLapack.factor] drop tolerance: keep every entry the elimination produces. */
public const val NO_DROP: Double = 0.0

/** The largest magnitude anywhere in [rows], the scale both tolerances are relative to. */
private fun largestMagnitude(rows: Array<MutableIntDoubleMap>): Double {
    var largest = 0.0
    for (row in rows) row.forEach { _, value -> if (abs(value) > largest) largest = abs(value) }
    return largest
}

/** A pivot must be at least this fraction of its column's largest magnitude, so a fill-reducing choice never
 *  sacrifices numerical stability. */
private const val PIVOT_THRESHOLD = 0.1

/** Candidate-bearing columns to examine before settling for the best pivot found (Suhl and Suhl). */
private const val MAX_CANDIDATE_COLS = 4

private class CountBuckets(private val size: Int) {
    private val head = IntArray(size + 2) { -1 }
    private val next = IntArray(size) { -1 }
    private val previous = IntArray(size) { -1 }

    /** No occupied count is below this. A hint rather than a fact, lowered by [add] and raised by lookups. */
    private var lowest = size + 1

    fun add(item: Int, count: Int) {
        val first = head[count]
        next[item] = first
        previous[item] = -1
        if (first != -1) previous[first] = item
        head[count] = item
        if (count < lowest) lowest = count
    }

    fun remove(item: Int, count: Int) {
        val before = previous[item]
        val after = next[item]
        if (before == -1) head[count] = after else next[before] = after
        if (after != -1) previous[after] = before
        previous[item] = -1
        next[item] = -1
    }

    fun moveTo(item: Int, from: Int, to: Int) {
        remove(item, from)
        add(item, to)
    }

    fun firstAt(count: Int): Int = head[count]

    fun after(item: Int): Int = next[item]

    /** The smallest occupied count at least [from], or -1 when every bucket from there up is empty. */
    fun smallestFrom(from: Int): Int {
        var count = if (from > lowest) from else lowest
        while (count <= size && head[count] == -1) count++
        if (count > size) return -1
        if (from <= lowest) lowest = count
        return count
    }
}

/**
 * @param u the row maps being eliminated.
 * @param m the dimension of the square matrix.
 * @param negligible a zero test: a value below it cannot be told from roundoff and is no pivot candidate.
 * @param dropBelow a fill control: the value is real, and discarding it trades accuracy for sparsity.
 */
private class MarkowitzState(
    private val u: Array<MutableIntDoubleMap>,
    private val m: Int,
    private val negligible: Double,
    private val dropBelow: Double,
) {
    private val rowActive = BooleanArray(m) { true }
    private val colActive = BooleanArray(m) { true }
    private val rowCount = IntArray(m)
    private val colCount = IntArray(m)

    /** The active rows holding a nonzero in each column. */
    private val colRows = Array(m) { MutableIntSet() }

    // The candidate column selectPivot() is scanning, gathered once and read twice.
    private val candidateRows = IntArray(m)
    private val candidateAbs = DoubleArray(m)

    // Active columns by their count, and active rows by theirs. Only positive counts are ever in them.
    private val columnsByCount = CountBuckets(m)
    private val rowsByCount = CountBuckets(m)

    /** The pivot chosen by the last successful [selectPivot]. */
    var pivotRow = -1
        private set
    var pivotCol = -1
        private set

    init {
        for (i in 0 until m) {
            rowCount[i] = u[i].size
            u[i].forEach { c, _ ->
                colCount[c]++
                colRows[c].add(i)
            }
        }
        for (i in 0 until m) if (rowCount[i] > 0) rowsByCount.add(i, rowCount[i])
        for (c in 0 until m) if (colCount[c] > 0) columnsByCount.add(c, colCount[c])
    }

    /**
     * Change a row's count and keep its bucket membership true. A count of zero is not represented, since an
     * empty row is no longer a pivot candidate.
     */
    private fun changeRowCount(i: Int, delta: Int) {
        val before = rowCount[i]
        val after = before + delta
        rowCount[i] = after
        if (!rowActive[i]) return
        when {
            before > 0 && after > 0 -> rowsByCount.moveTo(i, before, after)
            before > 0 -> rowsByCount.remove(i, before)
            after > 0 -> rowsByCount.add(i, after)
        }
    }

    /** [changeRowCount] for a column. */
    private fun changeColumnCount(c: Int, delta: Int) {
        val before = colCount[c]
        val after = before + delta
        colCount[c] = after
        if (!colActive[c]) return
        when {
            before > 0 && after > 0 -> columnsByCount.moveTo(c, before, after)
            before > 0 -> columnsByCount.remove(c, before)
            after > 0 -> columnsByCount.add(c, after)
        }
    }

    /** Pick the numerically acceptable pivot of least Markowitz count. False when none exists, so the
     *  remaining submatrix is singular. */
    @Suppress("CyclomaticComplexMethod", "NestedBlockDepth")
    fun selectPivot(): Boolean {
        pivotRow = -1
        pivotCol = -1
        var bestMark = Long.MAX_VALUE
        var bestAbs = 0.0
        val smallestRow = rowsByCount.smallestFrom(1)
        val minRow = if (smallestRow == -1) Int.MAX_VALUE else smallestRow
        var candidateCols = 0
        var r = columnsByCount.smallestFrom(1)
        while (r != -1) {
            var c = columnsByCount.firstAt(r)
            while (c != -1) {
                var colMax = 0.0
                var candidates = 0
                colRows[c].forEach { i ->
                    val a = abs(u[i].getOrDefault(c, 0.0))
                    candidateRows[candidates] = i
                    candidateAbs[candidates] = a
                    candidates++
                    if (a > colMax) colMax = a
                }
                if (colMax > negligible) {
                    val threshold = PIVOT_THRESHOLD * colMax
                    for (t in 0 until candidates) {
                        val a = candidateAbs[t]
                        if (a > negligible && a >= threshold) {
                            val i = candidateRows[t]
                            val mark = (rowCount[i] - 1).toLong() * (r - 1).toLong()
                            if (mark < bestMark || (mark == bestMark && a > bestAbs)) {
                                bestMark = mark
                                bestAbs = a
                                pivotRow = i
                                pivotCol = c
                            }
                        }
                    }
                    candidateCols++
                }
                // Every unscanned column has a count of at least r, so both bounds hold mid-class too.
                if (pivotRow != -1 &&
                    (candidateCols >= MAX_CANDIDATE_COLS || bestMark <= r.toLong() * (minRow - 1).toLong())
                ) {
                    return true
                }
                c = columnsByCount.after(c)
            }
            r = columnsByCount.smallestFrom(r + 1)
        }
        return pivotRow != -1
    }

    /** Eliminate the selected pivot, recording the column's multipliers into [l] and keeping all counts and
     *  the `colRows` index exact as fill-in appears and entries cancel. */
    @Suppress("NestedBlockDepth")
    fun eliminate(l: MutableIntDoubleMap) {
        val pRow = pivotRow
        val pCol = pivotCol
        // Unlinked before being marked dead, since the helpers below leave inactive items alone.
        if (rowCount[pRow] > 0) rowsByCount.remove(pRow, rowCount[pRow])
        if (colCount[pCol] > 0) columnsByCount.remove(pCol, colCount[pCol])
        rowActive[pRow] = false
        colActive[pCol] = false
        val pivotRowMap = u[pRow]
        // The pivot row leaves the active submatrix, so unindex it from its still-active columns.
        pivotRowMap.forEach { c, _ ->
            if (colActive[c]) {
                colRows[c].remove(pRow)
                changeColumnCount(c, -1)
            }
        }
        val pivot = pivotRowMap.getOrDefault(pCol, 0.0)
        // Only rows indexed under the pivot column carry an entry to eliminate.
        colRows[pCol].forEach { i ->
            if (i != pRow) {
                val target = u[i]
                val pSlot = target.slotOf(pCol)
                val f = (if (pSlot >= 0) target.valueAt(pSlot) else 0.0) / pivot
                l.put(i, f)
                if (pSlot >= 0) target.removeAt(pSlot)
                changeRowCount(i, -1)
                pivotRowMap.forEach { col, value ->
                    if (colActive[col]) { // skips the pivot column and already-pivoted columns
                        val slot = target.slotOf(col)
                        if (slot >= 0) {
                            val nv = target.valueAt(slot) - f * value
                            if (nv == 0.0 || abs(nv) < dropBelow) {
                                target.removeAt(slot)
                                changeRowCount(i, -1)
                                changeColumnCount(col, -1)
                                colRows[col].remove(i)
                            } else {
                                target.setValueAt(slot, nv)
                            }
                        } else {
                            val nv = -f * value
                            if (nv != 0.0 && abs(nv) >= dropBelow) {
                                target.put(col, nv)
                                changeRowCount(i, 1)
                                changeColumnCount(col, 1)
                                colRows[col].add(i)
                            }
                        }
                    }
                }
            }
        }
    }
}
