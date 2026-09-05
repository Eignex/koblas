package com.eignex.koblas.sparse.factorization.lu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.internal.MutableIntDoubleMap
import kotlin.math.*

/**
 * Sparse LU factorization `P·B·Q = L·U` of an `m × m` matrix with Markowitz threshold pivoting. The factors
 * are held in both orientations, indexed by pivot position.
 */
public class F64SparseMarkowitzLu private constructor(
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
     *  1.0 when equilibration is off. The solves correct for it. */
    private val rowScale: DoubleArray,
    /** Nonzeros in L and U including the diagonal, the factorization's fill. */
    override val nnz: Int,
) : F64SparseLuFactorization {

    private val solveAllocation = AllocationCapability(
        AllocationGuarantee.NO_MANAGED_OR_NATIVE,
        listOf(ScratchRequirement(ScratchKind.F64, m, count = 2)),
    )

    override val n: Int get() = m

    /**
     * Unit lower triangular, built from the by-column factors this holds. The diagonal is implicit in the
     * factorization and stored here, so `P·A·Q = L·U` reads without a special case for it.
     */
    override val l: F64SparseMatrix
        get() = triangular(lColIdx, lColVal, diagonalFirst = true) { 1.0 }

    /** Upper triangular, the diagonal taken from the pivots. */
    override val u: F64SparseMatrix
        get() = triangular(uColIdx, uColVal, diagonalFirst = false) { uDiag[it] }

    override val rowOrder: IntArray get() = perm.copyOf()

    override val columnOrder: IntArray get() = colPerm.copyOf()

    override val rowScaling: DoubleArray get() = rowScale.copyOf()

    /**
     * One triangle as CSC with rows ascending. The off-diagonal entries of a column are held unordered, so
     * they are sorted here; the diagonal is placed at the end it belongs to rather than sorted into them.
     */
    private inline fun triangular(
        columns: Array<IntArray>,
        values: Array<DoubleArray>,
        diagonalFirst: Boolean,
        diagonal: (Int) -> Double,
    ): F64SparseMatrix {
        val colPtr = IntArray(m + 1)
        for (k in 0 until m) colPtr[k + 1] = colPtr[k] + columns[k].size + 1
        val rowIdx = IntArray(colPtr[m])
        val entries = DoubleArray(colPtr[m])
        for (k in 0 until m) {
            val column = columns[k]
            var slot = colPtr[k]
            if (diagonalFirst) {
                rowIdx[slot] = k
                entries[slot] = diagonal(k)
                slot++
            }
            // Written into the destination and sorted there, rather than ordering the positions first: an
            // ordering of the indices costs a comparator, a boxed list and an unboxing per entry, and these
            // are get() properties, so a caller reading l twice pays for it twice. Insertion sort because a
            // factor column is short and already nearly ordered.
            val from = slot
            for (position in column.indices) {
                rowIdx[slot] = column[position]
                entries[slot] = values[k][position]
                slot++
            }
            sortColumn(rowIdx, entries, from, slot)
            if (!diagonalFirst) {
                rowIdx[slot] = k
                entries[slot] = diagonal(k)
            }
        }
        return F64SparseMatrix.wrap(m, m, colPtr, rowIdx, entries)
    }

    /**
     * Orders `[from, until)` of one column by row index, carrying each value with its index.
     *
     * Insertion sort rather than a library sort: what needs ordering is two parallel primitive arrays at
     * once, which no sort here takes, and a factor column is short enough that the quadratic term does not
     * bite.
     */
    private fun sortColumn(rowIdx: IntArray, entries: DoubleArray, from: Int, until: Int) {
        for (i in from + 1 until until) {
            val row = rowIdx[i]
            val value = entries[i]
            var j = i - 1
            while (j >= from && rowIdx[j] > row) {
                rowIdx[j + 1] = rowIdx[j]
                entries[j + 1] = entries[j]
                j--
            }
            rowIdx[j + 1] = row
            entries[j + 1] = value
        }
    }

    /** Always [NOT_SINGULAR]: a [F64SparseMarkowitzLu] only exists for a matrix that factored completely. */
    override val failedAt: Int get() = NOT_SINGULAR

    override val rcond: Double
        get() {
            if (m == 0) return 1.0
            var minimum = Double.POSITIVE_INFINITY
            var maximum = 0.0
            for (pivot in uDiag) {
                val magnitude = abs(pivot)
                minimum = minOf(minimum, magnitude)
                maximum = maxOf(maximum, magnitude)
            }
            return if (maximum == 0.0) 0.0 else minimum / maximum
        }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability = solveAllocation

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
        // Left uncleared: y reads only below k and xp only above k, so a borrowed buffer's contents are dead.
        workspace.borrow(m) { y ->
            workspace.borrow(m) { xp ->
                // L y = P (E b), in pivot-position space.
                sweep(descending = false, lRowIdx, lRowVal, y, skipDiagonal = false, divide = false) { k ->
                    b[perm[k]] * rowScale[perm[k]]
                }
                // U x' = y, with x' in pivot-column space. Entry 0 of each U row is the diagonal.
                sweep(descending = true, uRowIdx, uRowVal, xp, skipDiagonal = true, divide = true) { k -> y[k] }
                // x = Q x'. Safe in place, since xp is separate storage from out.
                for (k in 0 until m) out[colPerm[k]] = xp[k]
            }
        }
        return out
    }

    /** The transposed sweep of [solveInto]: `Uᵀ Lᵀ (P x) = Q b`. `b` is indexed by original column, the
     *  result by original row. */
    private fun btranInto(b: DoubleArray, out: DoubleArray, workspace: Workspace? = null): DoubleArray {
        requireShape(b.size == m) { "btran: b size ${b.size} != $m" }
        requireShape(out.size == m) { "btran: out size ${out.size} != $m" }
        // Left uncleared for the reason [ftranInto] gives. z ascends, w descends, each written before read.
        workspace.borrow(m) { z ->
            workspace.borrow(m) { w ->
                // Uᵀ z = Qᵀ b, forward and lower.
                sweep(descending = false, uColIdx, uColVal, z, skipDiagonal = false, divide = true) { k ->
                    b[colPerm[k]]
                }
                // Lᵀ w = z, back, upper, unit diagonal.
                sweep(descending = true, lColIdx, lColVal, w, skipDiagonal = false, divide = false) { k -> z[k] }
                // x = E·x', undoing the row equilibration.
                for (k in 0 until m) out[perm[k]] = w[k] * rowScale[perm[k]]
            }
        }
        return out
    }

    /**
     * One triangular sweep of a solve: at each pivot position, subtract the factor entries from what [init]
     * supplies and write the result into [x], which is also what the inner loop reads. The four sweeps of the
     * two directions differ only in these arguments.
     *
     * Inlined deliberately, and the direction is a flag rather than an `IntProgression` because a progression
     * parameter is a real object where a `for` over a range is a counted loop, which
     * `AllocationFreeTest` catches. Every flag here is a constant at every call site, so each site compiles
     * to the inner loop a hand-written sweep would.
     *
     * @param descending true to run from the last pivot position down, false to run up from the first.
     * @param idx the factor's index array per pivot position.
     * @param vals the factor's values, parallel to [idx].
     * @param x the destination, which the inner loop also reads at positions already written.
     * @param skipDiagonal true for a factor that stores its diagonal first, so entry 0 is not an update.
     * @param divide true when the pivot is not implicitly one and the result needs [uDiag].
     * @param init the right-hand side at a pivot position, before any factor entry is subtracted.
     */
    @Suppress("LongParameterList") // the five things the four sweeps vary, plus the destination
    private inline fun sweep(
        descending: Boolean,
        idx: Array<IntArray>,
        vals: Array<DoubleArray>,
        x: DoubleArray,
        skipDiagonal: Boolean,
        divide: Boolean,
        init: (Int) -> Double,
    ) {
        val first = if (skipDiagonal) 1 else 0
        val step = if (descending) -1 else 1
        val stop = if (descending) -1 else m
        var k = if (descending) m - 1 else 0
        while (k != stop) {
            var s = init(k)
            val ix = idx[k]
            val v = vals[k]
            for (t in first until ix.size) s -= v[t] * x[ix[t]]
            x[k] = if (divide) s / uDiag[k] else s
            k += step
        }
    }

    /** Entry points for factorizing. */
    public companion object {

        /**
         * Factorize the square [a], the implementation behind [F64SparseDecompositions.factor]. Returns a
         * [F64SingularSparseFactorization] when no acceptable pivot remains.
         */
        internal fun factorCsc(
            a: F64SparseMatrix,
            equilibrate: Boolean = false,
            dropTolerance: Double = NO_DROP,
        ): F64SparseLuFactorization {
            requireSquare(a, "F64SparseMarkowitzLu")
            // Sized from the matrix so no row map rehashes on the way in: the default holds nine entries,
            // and every row past that reallocated its table and both parallel arrays.
            val rowCount = IntArray(a.rows)
            for (j in 0 until a.cols) a.forEachInColumn(j) { i, _ -> rowCount[i]++ }
            val rows = Array(a.rows) { MutableIntDoubleMap(rowCount[it]) }
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
        ): F64SparseLuFactorization {
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
                    // A row below 2^-1023 needs a factor no double holds, and an infinite one would scale
                    // the row to infinities and take the tolerances below with it, since both are fractions
                    // of the largest magnitude present. Two steps would not help: rowScale has to stay
                    // finite for btran to undo it.
                    if (e == 1.0 || !e.isFinite()) continue
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
                if (!state.selectPivot()) return F64SingularSparseFactorization(m, failedAt = k)
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
        ): F64SparseMarkowitzLu {
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
            return F64SparseMarkowitzLu(
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
            // A counting transpose rather than a map per row. Step j contributes to row invPerm[origRow]
            // and j runs ascending, so each row receives its steps already in order: building a hash map
            // to sort the keys back into that order and then probe once more for every value is three
            // passes spent recovering what the input already had, plus m maps and m sorts.
            val counts = IntArray(m)
            for (j in 0 until m) lAtStep[j].forEach { origRow, _ -> counts[invPerm[origRow]]++ }
            val idx = Array(m) { IntArray(counts[it]) }
            val values = Array(m) { DoubleArray(counts[it]) }
            val filled = IntArray(m)
            for (j in 0 until m) {
                lAtStep[j].forEach { origRow, f ->
                    val k = invPerm[origRow]
                    val slot = filled[k]++
                    idx[k][slot] = j
                    values[k][slot] = f
                }
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

private fun inverseOf(permutation: IntArray): IntArray =
    IntArray(permutation.size).also { inverse -> for (i in permutation.indices) inverse[permutation[i]] = i }
