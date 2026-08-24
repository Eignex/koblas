package com.eignex.koblas.sparse.factorization.lu

import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.internal.IntBuckets
import com.eignex.koblas.sparse.internal.MutableIntDoubleMap
import com.eignex.koblas.sparse.internal.MutableIntSet
import kotlin.math.abs

/** One factor's entries indexed by column, parallel per column. */
internal class ColumnOrientation(val indices: Array<IntArray>, val values: Array<DoubleArray>)

/**
 * A magnitude this far below the matrix's largest is treated as zero rather than as a pivot candidate. This
 * is a zero test, relative rather than absolute; stability is [PIVOT_THRESHOLD]'s job.
 */
internal const val NEGLIGIBLE = 1e-14

/** The default [F64SparseLu.factor] drop tolerance: keep every entry the elimination produces. */
public const val NO_DROP: Double = 0.0

/** The largest magnitude anywhere in [rows], the scale both tolerances are relative to. */
internal fun largestMagnitude(rows: Array<MutableIntDoubleMap>): Double {
    var largest = 0.0
    for (row in rows) row.forEach { _, value -> if (abs(value) > largest) largest = abs(value) }
    return largest
}

/** A pivot must be at least this fraction of its column's largest magnitude, so a fill-reducing choice never
 *  sacrifices numerical stability. */
internal const val PIVOT_THRESHOLD = 0.1

/** Candidate-bearing columns to examine before settling for the best pivot found (Suhl and Suhl). */
internal const val MAX_CANDIDATE_COLS = 4

/**
 * @param u the row maps being eliminated.
 * @param m the dimension of the square matrix.
 * @param negligible a zero test: a value below it cannot be told from roundoff and is no pivot candidate.
 * @param dropBelow a fill control: the value is real, and discarding it trades accuracy for sparsity.
 */
internal class MarkowitzState(
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
    private val columnsByCount = IntBuckets(m, m + 1, m + 1)
    private val rowsByCount = IntBuckets(m, m + 1, m + 1)

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
    private fun changeRowCount(i: Int, delta: Int) = changeCount(rowCount, rowActive, rowsByCount, i, delta)

    /** [changeRowCount] for a column. */
    private fun changeColumnCount(c: Int, delta: Int) = changeCount(colCount, colActive, columnsByCount, c, delta)

    /** The shared body of [changeRowCount] and [changeColumnCount], which differ only in which axis they walk. */
    private fun changeCount(counts: IntArray, active: BooleanArray, buckets: IntBuckets, i: Int, delta: Int) {
        val before = counts[i]
        val after = before + delta
        counts[i] = after
        if (!active[i]) return
        when {
            before > 0 && after > 0 -> buckets.moveTo(i, before, after)
            before > 0 -> buckets.remove(i, before)
            after > 0 -> buckets.add(i, after)
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
                // One class loose on purpose. An unscanned column holds at least r entries and its best row
                // at least minRow, so its floor is (r - 1) * (minRow - 1) and one still in this class can
                // match the pivot in hand; this compares against the next class's floor instead. Scanning on
                // to rule that out lengthens every search and buys no fill, so the cheaper stop wins and the
                // Markowitz minimum here is approximate rather than exact.
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
                // colRows(pCol) indexes exactly the active rows holding an entry there, and every earlier
                // step removed its own pivot column from every row it touched, so the slot is always
                // present. The count below depends on that, so reading it unguarded keeps the two in step.
                val pSlot = target.slotOf(pCol)
                val f = target.valueAt(pSlot) / pivot
                l.put(i, f)
                target.removeAt(pSlot)
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
