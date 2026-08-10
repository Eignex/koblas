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
 * Sparse LU factorization `P·B·Q = L·U` of an `m × m` matrix, with **Markowitz threshold pivoting**: at
 * each step the pivot (near-)minimises fill (the Markowitz count `(rowNnz−1)·(colNnz−1)` over the active
 * submatrix, searched over a bounded set of lowest-count candidate columns à la Suhl & Suhl) among
 * entries that are numerically acceptable (`|a| ≥ τ·max|column|`), so the factors stay sparse instead of
 * filling toward `O(m²)`. Both a row permutation `P` and a column permutation `Q` are
 * produced; only the nonzeros of `L`/`U` are stored, so memory is `O(nnz)`.
 *
 * Right-looking Gaussian elimination over per-row hash maps; the factors are frozen into sparse arrays
 * in both orientations (indexed by **pivot position**) so both solve directions are `O(nnz)` triangular
 * solves. The forward direction's result is scattered back to original-column order by `Q`; the
 * transposed direction's right-hand side is gathered by `Q`.
 *
 * The portable [SparseFactorization]. Produced by [SparseLapack.factor] rather than constructed directly,
 * so a caller that holds the interface can be handed a host solver's factors instead without changing.
 * A singular matrix yields [SingularSparseFactorization], never an instance of this class — every
 * `SparseLu` is a complete factorization, which is why [failedAt] is constantly [NOT_SINGULAR].
 */
public class SparseLu private constructor(
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
     *  when equilibration is off. the forward sweep/the transposed sweep/[determinant] correct for it transparently. */
    private val rowScale: DoubleArray,
    /** Total nonzeros in `L` + `U` (incl. diagonal) — the factorization's fill. */
    override val nnz: Int,
) : SparseFactorization {

    override val n: Int get() = m

    /** Always false: a [SparseLu] only exists for a matrix that factored completely. */
    override val failedAt: Int get() = NOT_SINGULAR

    /**
     * Solve `B x = b`, or `Bᵀ x = b` when [transpose], into [out].
     *
     * The two directions are a simplex's FTRAN and BTRAN, and were spelled `ftran`/`btran` here until the
     * sparse and dense halves were made to agree on one vocabulary: this is the same operation the dense
     * `Lapack.solve` performs, so it carries the same name and the same transpose flag rather than a
     * private one. `b` is indexed by original row and the result by original column.
     */
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        if (transpose) btranInto(b, out, workspace) else ftranInto(b, out, workspace)

    /**
     * The forward direction of [solveInto]: `L U (Qᵀ x) = P (E b)`.
     *
     * Private because the seam speaks `solve`; the FTRAN name is kept here, where it names which of the two
     * triangular sweeps this is rather than standing in for "solve".
     */
    private fun ftranInto(b: DoubleArray, out: DoubleArray, workspace: Workspace? = null): DoubleArray {
        requireShape(b.size == m) { "ftran: b size ${b.size} != $m" }
        requireShape(out.size == m) { "ftran: out size ${out.size} != $m" }
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

    /** The transposed sweep of [solveInto]: `Uᵀ Lᵀ (P x) = Q b`. `b` is indexed by original column, the
     *  result by original row. Private for the reason [ftranInto] is. */
    private fun btranInto(b: DoubleArray, out: DoubleArray, workspace: Workspace? = null): DoubleArray {
        requireShape(b.size == m) { "solve: b size ${b.size} != $m" }
        requireShape(out.size == m) { "solve: out size ${out.size} != $m" }
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
    override fun determinant(): Double {
        var d = permutationSign(perm) * permutationSign(colPerm)
        for (k in 0 until m) d *= uDiag[k]
        for (i in 0 until m) d /= rowScale[i] // undo the row equilibration (rowScale all 1.0 when off)
        return d
    }

    /** Factorization entrypoints for [SparseLu]. */
    public companion object {

        /**
         * Factorize the square [a], the implementation behind [SparseLapack.factor].
         *
         * Returns a [SingularSparseFactorization] rather than null when no acceptable pivot remains: a
         * caller holding the interface gets the failing pivot position, which a null cannot carry, and the
         * dense side reports the same condition the same way.
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
         * Factorize the matrix whose rows are [rows] (`rows[i][col] = B[i][col]`, primitive per-row maps),
         * size [m].
         *
         * Internal, and takes [MutableIntDoubleMap] rather than a boxed `HashMap<Int, Double>` because the
         * elimination probes these maps once per entry per step. Build a [SparseMatrix] and use [factorCsc].
         *
         * **Consumes [rows]:** the maps are eliminated in place (and rescaled when [equilibrate]), so the
         * caller must not reuse them afterward.
         *
         * With [equilibrate], each row is first scaled by a power of two `eᵢ ≈ 1/max|rowᵢ|` (exact in
         * floating point) so pivoting is better conditioned; the scale is undone transparently in the
         * solves and [determinant].
         *
         * [dropTolerance] is a fraction of the largest magnitude in the matrix, applied after any
         * equilibration; see [SparseLapack.factor].
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
            val u = rows // eliminated in place; u[perm[k]] ends as U's pivot row k (pivot cols ≥ k)
            // Both tolerances are fractions of the largest magnitude present, so a matrix and the same
            // matrix scaled by 1e-8 factor identically. Measured after equilibration, which changes it.
            val scale = largestMagnitude(u)
            // L multipliers recorded per elimination step, keyed by the eliminated original row.
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

/**
 * A magnitude this far below the matrix's largest is treated as zero rather than as a pivot candidate.
 *
 * Relative, not absolute: an absolute floor would declare a perfectly well-conditioned matrix whose entries
 * all sit near `1e-10` singular at the first step, and would leave a matrix scaled up by `1e10` with no
 * protection at all. Set near double precision so it rejects only what cannot be told from roundoff — it is
 * a zero test, and stability is [PIVOT_THRESHOLD]'s job.
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

/** Markowitz stability threshold: a pivot must be at least this fraction of its column's largest
 *  magnitude, so fill-reducing choices never sacrifice numerical stability. */
private const val PIVOT_THRESHOLD = 0.1

/** Candidate-bearing columns to examine before settling for the best pivot found (Suhl & Suhl). */
private const val MAX_CANDIDATE_COLS = 4

/**
 * Items grouped by a small integer count, in doubly linked lists, with the smallest occupied count tracked.
 *
 * The Markowitz search needs two questions answered on every one of the `m` elimination steps: which columns
 * have the fewest entries, and what the smallest row count is. Answered by scanning, each costs `O(m)` per
 * step and so `O(m²)` over the factorization — which is the whole cost on a matrix that barely fills, where
 * the elimination itself does almost nothing. Answered by this, each costs constant time per *change*, and the
 * changes are exactly the count updates the elimination already performs.
 *
 * Doubly linked because items leave from the middle: a column whose count drops has to be unlinked from
 * wherever it sits, not found first. [smallestFrom] advances its hint monotonically within a step, so a search
 * that walks several count classes pays for each once.
 */
private class CountBuckets(private val size: Int) {
    private val head = IntArray(size + 2) { -1 }
    private val next = IntArray(size) { -1 }
    private val previous = IntArray(size) { -1 }

    /** No occupied count is below this. A hint rather than a fact: [add] lowers it, lookups raise it. */
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
 *
 * The two magnitude limits are separate because they answer different questions. [negligible] is a zero
 * test: below it a value cannot be told from roundoff, so it is not a pivot candidate and its column may be
 * structurally singular. [dropBelow] is a fill control: the value is real, and discarding it trades accuracy
 * for sparsity, which is a caller's decision and defaults to not happening.
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

    /** Per column: the set of active rows holding a nonzero in it (values unused). */
    private val colRows = Array(m) { MutableIntDoubleMap() }

    // Active columns by their count, and active rows by theirs, maintained as the counts change rather
    // than rebuilt per step. Only items with a positive count are ever in them.
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
                colRows[c].put(i, 0.0)
            }
        }
        for (i in 0 until m) if (rowCount[i] > 0) rowsByCount.add(i, rowCount[i])
        for (c in 0 until m) if (colCount[c] > 0) columnsByCount.add(c, colCount[c])
    }

    /**
     * Change a row's count and keep its bucket membership true.
     *
     * Every count update in [eliminate] goes through here or [changeColumnCount], which is what lets the
     * search trust the buckets instead of rescanning. A count of zero is not represented: an empty row is no
     * longer a pivot candidate, and leaving it in a bucket would mean filtering it out on every lookup.
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

    /** Pick the minimum-Markowitz-count numerically acceptable pivot; false if none exists (singular). */
    @Suppress("CyclomaticComplexMethod", "NestedBlockDepth")
    fun selectPivot(): Boolean {
        pivotRow = -1
        pivotCol = -1
        var bestMark = Long.MAX_VALUE
        var bestAbs = 0.0
        // Both of these were O(m) scans per step, and on a matrix that hardly fills they were the entire
        // cost of the factorization: O(m²) of bookkeeping around an elimination doing almost nothing.
        val smallestRow = rowsByCount.smallestFrom(1)
        val minRow = if (smallestRow == -1) Int.MAX_VALUE else smallestRow
        var candidateCols = 0
        var r = columnsByCount.smallestFrom(1)
        while (r != -1) {
            var c = columnsByCount.firstAt(r)
            while (c != -1) {
                var colMax = 0.0
                colRows[c].forEach { i, _ ->
                    val a = abs(u[i].getOrDefault(c, 0.0))
                    if (a > colMax) colMax = a
                }
                if (colMax > negligible) {
                    val threshold = PIVOT_THRESHOLD * colMax
                    colRows[c].forEach { i, _ ->
                        val a = abs(u[i].getOrDefault(c, 0.0))
                        if (a > negligible && a >= threshold) {
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
                if (colMax > negligible) candidateCols++
                // Checked inside the walk, not after it. Every unscanned column has a count of at least `r`,
                // here or in a later class, so both bounds hold as well here as at the class boundary — and
                // finishing the class first is what left this O(m) per step on a matrix whose columns share
                // one count. A diagonal is the extreme case: one class holds every column, so the search
                // walked the whole matrix to choose each of its m pivots.
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

    /** Eliminate the selected pivot, recording the column's multipliers into [l] and keeping all
     *  counts and the `colRows` index exact as fill-in appears and entries cancel. */
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
        // The pivot row leaves the active submatrix: unindex it from its still-active columns.
        pivotRowMap.forEach { c, _ ->
            if (colActive[c]) {
                colRows[c].remove(pRow)
                changeColumnCount(c, -1)
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
                changeRowCount(i, -1)
                pivotRowMap.forEach { col, value ->
                    if (colActive[col]) { // skips the pivot column and already-pivoted columns
                        val slot = target.slotOf(col)
                        if (slot >= 0) {
                            val nv = target.valueAt(slot) - f * value
                            if (nv == 0.0 || abs(nv) < dropBelow) {
                                target.remove(col)
                                changeRowCount(i, -1)
                                changeColumnCount(col, -1)
                                colRows[col].remove(i)
                            } else {
                                target.put(col, nv)
                            }
                        } else {
                            val nv = -f * value
                            if (nv != 0.0 && abs(nv) >= dropBelow) {
                                target.put(col, nv)
                                changeRowCount(i, 1)
                                changeColumnCount(col, 1)
                                colRows[col].put(i, 0.0)
                            }
                        }
                    }
                }
            }
        }
    }
}
