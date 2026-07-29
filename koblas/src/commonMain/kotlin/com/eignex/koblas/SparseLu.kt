package com.eignex.koblas

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.pow

/**
 * Sparse LU factorization `P·B·Q = L·U` of an `m × m` matrix, with **Markowitz threshold pivoting**: at
 * each step the pivot (near-)minimises fill (the Markowitz count `(rowNnz−1)·(colNnz−1)` over the active
 * submatrix, searched over a bounded set of lowest-count candidate columns à la Suhl & Suhl) among
 * entries that are numerically acceptable (`|a| ≥ τ·max|column|`), so the factors stay sparse instead of
 * filling toward `O(m²)`. Both a row permutation `P` and a column permutation `Q` are
 * produced; only the nonzeros of `L`/`U` are stored, so memory is `O(nnz)`.
 *
 * Right-looking Gaussian elimination over per-row hash maps; the factors are frozen into sparse arrays
 * in both orientations (indexed by **pivot position**) so [ftran] (`B x = b`) and [btran] (`Bᵀ x = b`)
 * are `O(nnz)` triangular solves. [ftran]'s result is scattered back to original-column order by `Q`;
 * [btran]'s right-hand side is gathered by `Q`. [factorize] returns null on a (near-)singular matrix.
 */
class SparseLu private constructor(
    private val m: Int,
    private val perm: IntArray, // perm[k] = original row index now at pivot position k
    private val colPerm: IntArray, // colPerm[k] = original column index now at pivot position k
    private val lRowIdx: Array<IntArray>, // L by row (pivot positions < k; unit diagonal implicit)
    private val lRowVal: Array<DoubleArray>,
    private val uRowIdx: Array<IntArray>, // U by row (pivot positions ≥ k); first entry of row k is the diagonal
    private val uRowVal: Array<DoubleArray>,
    private val lColIdx: Array<IntArray>, // L by column (pivot positions > k)
    private val lColVal: Array<DoubleArray>,
    private val uColIdx: Array<IntArray>, // U by column (pivot positions < k, strictly upper)
    private val uColVal: Array<DoubleArray>,
    private val uDiag: DoubleArray,
    /** Per-original-row equilibration factor `eᵢ` applied before factorization (`L·U = E·B`); all 1.0
     *  when equilibration is off. [ftran]/[btran]/[determinant] correct for it transparently. */
    private val rowScale: DoubleArray,
    /** Total nonzeros in `L` + `U` (incl. diagonal) — the factorization's fill. */
    val nnz: Int,
) {

    /** Solve `B x = b` (FTRAN). `b` is indexed by original row; the result by original column.
     *  Allocates the result and two intermediates; [ftranInto] reuses caller-owned buffers instead. */
    fun ftran(b: DoubleArray): DoubleArray = ftranInto(b, DoubleArray(m))

    /**
     * Solve `B x = b` (FTRAN) into [out], which is returned. With a [workspace] the two pivot-space
     * intermediates are reused too, so a simplex iteration allocates nothing at all. [out] may be [b].
     */
    fun ftranInto(b: DoubleArray, out: DoubleArray, workspace: Workspace? = null): DoubleArray {
        require(b.size == m) { "ftran: b size ${b.size} != $m" }
        require(out.size == m) { "ftran: out size ${out.size} != $m" }
        val y = workspace?.take(m) ?: DoubleArray(m)
        val xp = workspace?.take(m) ?: DoubleArray(m)
        y.fill(0.0)
        xp.fill(0.0)
        // L y = P (E b) (forward); rows/cols are in pivot-position space.
        for (k in 0 until m) {
            var s = b[perm[k]] * rowScale[perm[k]]
            val idx = lRowIdx[k]
            val v = lRowVal[k]
            for (t in idx.indices) s -= v[t] * y[idx[t]]
            y[k] = s
        }
        // U x' = y (back); x' is in pivot-column space.
        for (k in m - 1 downTo 0) {
            var s = y[k]
            val idx = uRowIdx[k]
            val v = uRowVal[k]
            for (t in 1 until idx.size) s -= v[t] * xp[idx[t]] // skip [0] = diagonal
            xp[k] = s / uDiag[k]
        }
        // x = Q x'  ⇒  x[colPerm[k]] = x'[k]. Safe in place: xp is separate storage from out.
        for (k in 0 until m) out[colPerm[k]] = xp[k]
        workspace?.release(xp)
        workspace?.release(y)
        return out
    }

    /** Solve `Bᵀ x = b` (BTRAN). `b` is indexed by original column; the result by original row.
     *  Allocates; [btranInto] writes into a caller-owned destination. */
    fun btran(b: DoubleArray): DoubleArray = btranInto(b, DoubleArray(m))

    /** Solve `Bᵀ x = b` (BTRAN) into [out], which is returned. [out] may be [b]. */
    fun btranInto(b: DoubleArray, out: DoubleArray, workspace: Workspace? = null): DoubleArray {
        require(b.size == m) { "btran: b size ${b.size} != $m" }
        require(out.size == m) { "btran: out size ${out.size} != $m" }
        val z = workspace?.take(m) ?: DoubleArray(m)
        val w = workspace?.take(m) ?: DoubleArray(m)
        z.fill(0.0)
        w.fill(0.0)
        // Uᵀ z = Qᵀ b (forward, lower).
        for (k in 0 until m) {
            var s = b[colPerm[k]]
            val idx = uColIdx[k]
            val v = uColVal[k]
            for (t in idx.indices) s -= v[t] * z[idx[t]]
            z[k] = s / uDiag[k]
        }
        // Lᵀ w = z (back, upper, unit diagonal).
        for (k in m - 1 downTo 0) {
            var s = z[k]
            val idx = lColIdx[k]
            val v = lColVal[k]
            for (t in idx.indices) s -= v[t] * w[idx[t]]
            w[k] = s
        }
        // x = E·x' with x[perm[k]] = w[k]·e_{perm[k]}.
        for (k in 0 until m) out[perm[k]] = w[k] * rowScale[perm[k]]
        workspace?.release(w)
        workspace?.release(z)
        return out
    }

    /**
     * `det(B)` in floating point: `sign(P)·sign(Q)·∏ uDiag / ∏ eᵢ` — the factors are of `E·B`, so the
     * row-equilibration product is divided back out (a no-op when equilibration is off). For an integer
     * matrix the true determinant is an integer; this float value is only a *guess* of it.
     */
    fun determinant(): Double {
        var d = permutationSign(perm) * permutationSign(colPerm)
        for (k in 0 until m) d *= uDiag[k]
        for (i in 0 until m) d /= rowScale[i] // undo the row equilibration (rowScale all 1.0 when off)
        return d
    }

    /** Factorization entrypoints for [SparseLu]. */
    companion object {

        /** Factorize the `m × m` matrix [a] (CSC); convenience over [factorize] taking per-row maps. */
        fun factorize(a: SparseMatrix, equilibrate: Boolean = false): SparseLu? {
            require(a.rows == a.cols) { "SparseLu requires a square matrix; got ${a.rows}x${a.cols}" }
            val rows = Array(a.rows) { MutableIntDoubleMap() }
            for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> rows[i].put(j, v) }
            return factorize(rows, a.rows, equilibrate)
        }

        /**
         * Factorize the matrix whose rows are [rows] (`rows[i][col] = B[i][col]`, dense per-row maps),
         * size [m]. Returns null if no numerically acceptable pivot remains (singular matrix).
         * Convenience over the internal primitive-map path; the boxed maps are copied, not consumed.
         */
        fun factorize(rows: Array<HashMap<Int, Double>>, m: Int, equilibrate: Boolean = false): SparseLu? {
            val prim = Array(m) { i ->
                val map = MutableIntDoubleMap(rows[i].size)
                for ((c, v) in rows[i]) map.put(c, v)
                map
            }
            return factorize(prim, m, equilibrate)
        }

        /**
         * Factorize the matrix whose rows are [rows] (`rows[i][col] = B[i][col]`, primitive per-row
         * maps), size [m]. Returns null if no numerically acceptable pivot remains (singular matrix).
         *
         * **Consumes [rows]:** the maps are eliminated in place (and rescaled when [equilibrate]), so the
         * caller must not reuse them afterward.
         *
         * With [equilibrate], each row is first scaled by a power of two `eᵢ ≈ 1/max|rowᵢ|` (exact in
         * floating point) so pivoting is better conditioned; the scale is undone transparently in the
         * solves and [determinant].
         */
        internal fun factorize(rows: Array<MutableIntDoubleMap>, m: Int, equilibrate: Boolean = false): SparseLu? {
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
            val u = rows // eliminated in place; u[perm[k]] ends as U's pivot row k (pivot cols ≥ k)
            // L multipliers recorded per elimination step, keyed by the eliminated original row.
            val lAtStep = Array(m) { MutableIntDoubleMap() }
            val perm = IntArray(m) { -1 }
            val colPerm = IntArray(m) { -1 }
            val state = MarkowitzState(u, m)
            for (k in 0 until m) {
                if (!state.selectPivot()) return null // singular
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
            // U row k (pivot space): the pivot row's entries, original col → pivot position (all ≥ k),
            // diagonal (k) first then ascending.
            val uRowIdx = Array(m) { k -> sortedKeysOf(u[perm[k]]) { invColPerm[it] } }
            val uRowVal = Array(m) { k ->
                val row = u[perm[k]]
                DoubleArray(uRowIdx[k].size) { t -> row.getOrDefault(colPerm[uRowIdx[k][t]], 0.0) }
            }
            // L row k' (pivot space): multipliers from each step j < k' that eliminated row perm[k'].
            val lRowMap = Array(m) { MutableIntDoubleMap() }
            for (j in 0 until m) {
                lAtStep[j].forEach { origRow, f -> lRowMap[invPerm[origRow]].put(j, f) }
            }
            val lRowIdx = Array(m) { k -> sortedKeysOf(lRowMap[k]) { it } }
            val lRowVal = Array(m) { k ->
                DoubleArray(lRowIdx[k].size) { t -> lRowMap[k].getOrDefault(lRowIdx[k][t], 0.0) }
            }
            // Column orientations (pivot space): U strictly-upper by column, L by column.
            val uColB = Array(m) { ArrayList<Int>() }
            val uColBv = Array(m) { ArrayList<Double>() }
            for (k in 0 until m) {
                val idx = uRowIdx[k]
                val v = uRowVal[k]
                for (t in idx.indices) {
                    val col = idx[t]
                    if (col > k) {
                        uColB[col].add(k)
                        uColBv[col].add(v[t])
                    }
                }
            }
            val lColB = Array(m) { ArrayList<Int>() }
            val lColBv = Array(m) { ArrayList<Double>() }
            for (k in 0 until m) {
                val idx = lRowIdx[k]
                val v = lRowVal[k]
                for (t in idx.indices) {
                    lColB[idx[t]].add(k)
                    lColBv[idx[t]].add(v[t])
                }
            }
            var nnz = 0
            for (k in 0 until m) nnz += uRowIdx[k].size + lRowIdx[k].size
            return SparseLu(
                m, perm, colPerm, lRowIdx, lRowVal, uRowIdx, uRowVal,
                Array(m) { lColB[it].toIntArray() }, Array(m) { lColBv[it].toDoubleArray() },
                Array(m) { uColB[it].toIntArray() }, Array(m) { uColBv[it].toDoubleArray() },
                uDiag, rowScale, nnz,
            )
        }
    }
}

private const val TOL = 1e-9

/** Markowitz stability threshold: a pivot must be at least this fraction of its column's largest
 *  magnitude, so fill-reducing choices never sacrifice numerical stability. */
private const val PIVOT_THRESHOLD = 0.1

/** Candidate-bearing columns to examine before settling for the best pivot found (Suhl & Suhl). */
private const val MAX_CANDIDATE_COLS = 4

/**
 * Incrementally-maintained elimination state for the Markowitz factorization: active row/column flags,
 * per-row/per-column nonzero counts over the active submatrix, and a column → active-rows index
 * (`colRows`) so pivot search and elimination touch only the rows that actually hold an entry.
 *
 * [selectPivot] buckets active columns by count and scans count classes in ascending order, computing
 * each candidate column's max magnitude on demand for the stability threshold. The scan stops at a
 * class boundary once either bound holds:
 *
 * - exactness: any entry in an unscanned column has Markowitz count `≥ r · (minRowCount − 1)`, so a
 *   best-so-far at or below that is a true global minimum;
 * - bounded search (Suhl & Suhl): at least `MAX_CANDIDATE_COLS` candidate-bearing columns have been
 *   examined — the best of the lowest count classes is kept even if a marginally better pivot might
 *   exist in a higher class. This caps per-step search work; fill in practice grows only a couple of
 *   percent over the exact minimum, versus a full active-submatrix rescan per step.
 *
 * [eliminate] keeps every count and index exact as entries are created and cancelled.
 */
private class MarkowitzState(private val u: Array<MutableIntDoubleMap>, private val m: Int) {
    private val rowActive = BooleanArray(m) { true }
    private val colActive = BooleanArray(m) { true }
    private val rowCount = IntArray(m)
    private val colCount = IntArray(m)

    /** Per column: the set of active rows holding a nonzero in it (values unused). */
    private val colRows = Array(m) { MutableIntDoubleMap() }

    // Per-step buckets of active columns keyed by colCount (linked lists over these arrays).
    private val bucketHead = IntArray(m + 1)
    private val bucketNext = IntArray(m)

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
                colRows[c].put(i, 0.0)
            }
        }
    }

    /** Pick the minimum-Markowitz-count numerically acceptable pivot; false if none exists (singular). */
    @Suppress("CyclomaticComplexMethod", "NestedBlockDepth")
    fun selectPivot(): Boolean {
        pivotRow = -1
        pivotCol = -1
        var bestMark = Long.MAX_VALUE
        var bestAbs = 0.0
        var minRow = Int.MAX_VALUE
        for (i in 0 until m) {
            if (rowActive[i] && rowCount[i] in 1 until minRow) minRow = rowCount[i]
        }
        var maxCount = 0
        bucketHead.fill(-1)
        for (c in 0 until m) {
            val r = colCount[c]
            if (colActive[c] && r > 0) {
                bucketNext[c] = bucketHead[r]
                bucketHead[r] = c
                if (r > maxCount) maxCount = r
            }
        }
        var candidateCols = 0
        for (r in 1..maxCount) {
            var c = bucketHead[r]
            while (c != -1) {
                var colMax = 0.0
                colRows[c].forEach { i, _ ->
                    val a = abs(u[i].getOrDefault(c, 0.0))
                    if (a > colMax) colMax = a
                }
                if (colMax >= TOL) {
                    val threshold = PIVOT_THRESHOLD * colMax
                    colRows[c].forEach { i, _ ->
                        val a = abs(u[i].getOrDefault(c, 0.0))
                        if (a >= TOL && a >= threshold) {
                            val mark = (rowCount[i] - 1).toLong() * (r - 1).toLong()
                            if (mark < bestMark || (mark == bestMark && a > bestAbs)) {
                                bestMark = mark
                                bestAbs = a
                                pivotRow = i
                                pivotCol = c
                            }
                        }
                    }
                }
                if (colMax >= TOL) candidateCols++
                c = bucketNext[c]
            }
            if (pivotRow != -1 &&
                (candidateCols >= MAX_CANDIDATE_COLS || bestMark <= r.toLong() * (minRow - 1).toLong())
            ) {
                return true
            }
        }
        return pivotRow != -1
    }

    /** Eliminate the selected pivot, recording the column's multipliers into [l] and keeping all
     *  counts and the `colRows` index exact as fill-in appears and entries cancel. */
    @Suppress("NestedBlockDepth")
    fun eliminate(l: MutableIntDoubleMap) {
        val pRow = pivotRow
        val pCol = pivotCol
        rowActive[pRow] = false
        colActive[pCol] = false
        val pivotRowMap = u[pRow]
        // The pivot row leaves the active submatrix: unindex it from its still-active columns.
        pivotRowMap.forEach { c, _ ->
            if (colActive[c]) {
                colRows[c].remove(pRow)
                colCount[c]--
            }
        }
        val pivot = pivotRowMap.getOrDefault(pCol, 0.0)
        // Only rows indexed under the pivot column carry an entry to eliminate.
        colRows[pCol].forEach { i, _ ->
            if (i != pRow) {
                val target = u[i]
                val f = target.getOrDefault(pCol, 0.0) / pivot
                l.put(i, f)
                target.remove(pCol)
                rowCount[i]--
                pivotRowMap.forEach { col, value ->
                    if (colActive[col]) { // skips the pivot column and already-pivoted columns
                        val slot = target.slotOf(col)
                        if (slot >= 0) {
                            val nv = target.valueAt(slot) - f * value
                            if (abs(nv) < TOL) {
                                target.remove(col)
                                rowCount[i]--
                                colCount[col]--
                                colRows[col].remove(i)
                            } else {
                                target.put(col, nv)
                            }
                        } else {
                            val nv = -f * value
                            if (abs(nv) >= TOL) {
                                target.put(col, nv)
                                rowCount[i]++
                                colCount[col]++
                                colRows[col].put(i, 0.0)
                            }
                        }
                    }
                }
            }
        }
    }
}
