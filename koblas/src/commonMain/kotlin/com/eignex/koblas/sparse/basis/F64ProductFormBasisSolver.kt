package com.eignex.koblas.sparse.basis

import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireInBounds
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.factorization.lu.F64SparseMarkowitzLu
import com.eignex.koblas.sparse.factorization.lu.ReachableSolveScratch

/** Updates a portable solver folds in before advising a rebuild; past this the chain costs more than it saves. */
public const val DEFAULT_ETA_LIMIT: Int = 50

/**
 * The portable [F64BasisSolver]: a sparse LU of the basis at the last [refactorize], plus a chain of
 * elementary transforms, one per [update].
 *
 * This is the product form of the inverse. Replacing basis slot `p` with a column whose spike is `η` makes
 * the new basis `B·E`, where `E` is the identity with column `p` set to `η`; its inverse differs from the
 * identity in that column alone, so applying it and its transpose costs the spike's nonzeros. The spike
 * arrives already indexed, so only the entries it carries are stored, and a hypersparse pivot leaves a
 * correspondingly small transform behind.
 *
 * The chain lengthens the solves and accumulates rounding, so past [etaLimit] updates [update] answers
 * [BasisUpdate.REFACTORIZE]. The solves themselves run densely between the seam and the base
 * factorization, which is what the portable sparse LU offers; the seam stays indexed so a host binding that
 * solves hypersparsely has nothing to undo.
 *
 * @param a the matrix whose columns the basis is drawn from, fixed for this solver's lifetime.
 * @param lu the factorization backend the rebuilds go through.
 * @param etaLimit updates folded in before [update] starts asking for a rebuild.
 */

public class F64ProductFormBasisSolver(
    private val a: F64SparseMatrix,
    private val lu: F64SparseDecompositions,
    private val etaLimit: Int = DEFAULT_ETA_LIMIT,
) : F64BasisSolver {
    init {
        requireShape(a.rows <= a.cols) { "a basis needs ${a.rows} columns to choose from; a has ${a.cols}" }
        require(etaLimit > 0) { "etaLimit must be positive: $etaLimit" }
    }

    override val n: Int = a.rows

    private val basicIndex = IntArray(n)
    private var base: F64SparseFactorization? = null
    private var closed = false

    // The chain in update order, the pivot held apart from the spike so applying one is a loop over the
    // off-pivot entries and a single divide. The pivots sit in primitive arrays because every solve walks
    // the whole chain, and a boxed list costs an unbox per eta on that path.
    private var etaPivotRow = IntArray(etaLimit)
    private var etaPivot = DoubleArray(etaLimit)
    private var etaCount = 0
    private val etaIndices = ArrayList<IntArray>()
    private val etaValues = ArrayList<DoubleArray>()

    private val dense = DoubleArray(n)

    /** Scratch for the reachability-limited forward solve, allocated once so a solve allocates nothing. */
    private val reachScratch = ReachableSolveScratch(n)
    private val reachPivots = IntArray(n)
    private val reachResult = IntArray(n)
    private val reachSeen = IntArray(n) { -1 }
    private var reachStamp = 0
    private val workspace = Workspace().apply { reserve(n, count = 3) }

    override val updateCount: Int get() {
        checkOpen()
        return etaCount
    }

    override val singular: Boolean get() = base.let { it == null || it.singular }

    override val rcond: Double
        get() {
            checkOpen()
            val baseQuality = base?.rcond ?: return 0.0
            var smallest = 1.0
            var largest = 1.0
            for (j in 0 until etaCount) {
                val magnitude = kotlin.math.abs(etaPivot[j])
                smallest = minOf(smallest, magnitude)
                largest = maxOf(largest, magnitude)
            }
            return minOf(baseQuality, smallest / largest)
        }

    override val nnz: Int
        get() {
            checkOpen()
            var total = base?.nnz ?: 0
            for (entries in etaIndices) total += entries.size + 1
            return total
        }

    override fun refactorize(basicIndex: IntArray): Boolean {
        checkOpen()
        requireShape(basicIndex.size == n) { "refactorize: basicIndex size ${basicIndex.size} != $n" }
        for (t in 0 until n) requireInBounds(basicIndex[t], a.cols)
        // Factorized before any of this is committed, so a backend that throws rather than answering
        // singular leaves the solver on the basis it already holds instead of on one it cannot solve.
        val factors = lu.factor(basisMatrix(basicIndex))
        val previous = base
        basicIndex.copyInto(this.basicIndex)
        dropChain()
        base = factors
        previous?.close()
        return !factors.singular
    }

    /**
     * [expectedDensity] chooses the sweep: below [REACHABLE_FTRAN_MAX_DENSITY] the solve visits only the
     * positions the right-hand side can reach, above it the dense sweep is cheaper than tracking which.
     */
    @OptIn(UnsafeKoblasApi::class)
    override fun ftran(x: F64IndexedVector, expectedDensity: Double) {
        checkOpen()
        val factors = solvable(x)
        if (expectedDensity < REACHABLE_FTRAN_MAX_DENSITY && factors is F64SparseMarkowitzLu) {
            for (t in 0 until x.count) dense[x.indices[t]] = x.values[x.indices[t]]
            var produced = factors.ftranReachable(dense, x.indices, x.count, reachScratch, reachPivots, reachResult)
            produced = applyEtasTracking(produced)
            // Filled past the seam, which is what the count setter exists for: clear the old pattern, then
            // write the reachable one and take the values straight out of the dense buffer, clearing it as
            // it is read so neither side pays a pass over n.
            x.clear()
            for (t in 0 until produced) {
                val i = reachResult[t]
                x.indices[t] = i
                x.values[i] = dense[i]
                dense[i] = 0.0
            }
            x.count = produced
            return
        }
        x.gather(dense)
        factors.solveInto(dense, dense, transpose = false, workspace = workspace)
        for (j in 0 until etaCount) applyEta(j)
        x.scatter(dense)
    }

    /**
     * Applies the eta chain to [dense] while extending the result pattern, which [reachResult] carries in.
     *
     * Each eta is already sparse in its own right, but it can write positions the factorization's reach did
     * not name, so the pattern has to grow with it or those entries would be dropped on the way out. A stamp
     * array does the membership test in constant time per touched position.
     *
     * @return the number of positions [reachResult] now names.
     */
    private fun applyEtasTracking(reached: Int): Int {
        if (etaCount == 0) return reached
        val seen = reachSeen
        val stamp = ++reachStamp
        var count = reached
        for (t in 0 until reached) seen[reachResult[t]] = stamp
        for (j in 0 until etaCount) {
            val pivotRow = etaPivotRow[j]
            val scaled = dense[pivotRow] / etaPivot[j]
            if (scaled != 0.0) {
                val indices = etaIndices[j]
                val values = etaValues[j]
                for (k in indices.indices) {
                    val i = indices[k]
                    dense[i] -= scaled * values[k]
                    if (seen[i] != stamp) {
                        seen[i] = stamp
                        reachResult[count++] = i
                    }
                }
            }
            dense[pivotRow] = scaled
            if (seen[pivotRow] != stamp) {
                seen[pivotRow] = stamp
                reachResult[count++] = pivotRow
            }
        }
        return count
    }

    /** [expectedDensity] is not read, as in [ftran]. */
    override fun btran(x: F64IndexedVector, expectedDensity: Double) {
        checkOpen()
        val factors = solvable(x)
        x.gather(dense)
        for (j in etaCount - 1 downTo 0) applyEtaTransposed(j)
        factors.solveInto(dense, dense, transpose = true, workspace = workspace)
        x.scatter(dense)
    }

    override fun solveQuality(rhs: DoubleArray, solution: F64IndexedVector, transpose: Boolean): F64BasisSolveQuality {
        checkOpen()
        check(!singular) { "solveQuality: the basis is singular" }
        return basisSolveQuality(a, basicIndex, rhs, solution, transpose)
    }

    override fun update(
        pivotRow: Int,
        entering: Int,
        spike: F64IndexedVector,
        pivotEta: F64IndexedVector?,
    ): BasisUpdate {
        checkOpen()
        requireInBounds(pivotRow, n)
        requireInBounds(entering, a.cols)
        requireShape(spike.size == n) { "update: spike size ${spike.size} != $n" }
        if (singular) return BasisUpdate.SINGULAR
        val pivot = spike[pivotRow]
        if (pivot == 0.0 || !pivot.isFinite()) return BasisUpdate.SINGULAR

        var stored = 0
        spike.forEachStored { i, v -> if (i != pivotRow && v != 0.0) stored++ }
        val indices = IntArray(stored)
        val values = DoubleArray(stored)
        var at = 0
        spike.forEachStored { i, v ->
            if (i != pivotRow && v != 0.0) {
                indices[at] = i
                values[at] = v
                at++
            }
        }
        if (etaCount == etaPivotRow.size) {
            etaPivotRow = etaPivotRow.copyOf(etaCount * 2)
            etaPivot = etaPivot.copyOf(etaCount * 2)
        }
        etaPivotRow[etaCount] = pivotRow
        etaPivot[etaCount] = pivot
        etaCount++
        etaIndices.add(indices)
        etaValues.add(values)
        basicIndex[pivotRow] = entering
        return if (updateCount >= etaLimit) BasisUpdate.REFACTORIZE else BasisUpdate.APPLIED
    }

    /** The forward transform: divide at the pivot, then eliminate the spike's other entries. */
    private fun applyEta(j: Int) {
        val p = etaPivotRow[j]
        val scaled = dense[p] / etaPivot[j]
        if (scaled != 0.0) {
            val indices = etaIndices[j]
            val values = etaValues[j]
            for (k in indices.indices) dense[indices[k]] -= scaled * values[k]
        }
        dense[p] = scaled
    }

    /** The transposed transform: gather the spike against the vector, then divide at the pivot. */
    private fun applyEtaTransposed(j: Int) {
        val p = etaPivotRow[j]
        val indices = etaIndices[j]
        val values = etaValues[j]
        var acc = dense[p]
        for (k in indices.indices) acc -= values[k] * dense[indices[k]]
        dense[p] = acc / etaPivot[j]
    }

    /** The basis of [basicIndex] in CSC, its columns copied from [a] as they lie. */
    @OptIn(UnsafeKoblasApi::class)
    private fun basisMatrix(basicIndex: IntArray): F64SparseMatrix {
        val colPtr = IntArray(n + 1)
        for (t in 0 until n) {
            val j = basicIndex[t]
            colPtr[t + 1] = colPtr[t] + (a.colPtr[j + 1] - a.colPtr[j])
        }
        val rowIdx = IntArray(colPtr[n])
        val values = DoubleArray(colPtr[n])
        var at = 0
        for (t in 0 until n) {
            a.forEachInColumn(basicIndex[t]) { i, v ->
                rowIdx[at] = i
                values[at] = v
                at++
            }
        }
        return F64SparseMatrix.wrap(n, n, colPtr, rowIdx, values)
    }

    private fun dropChain() {
        etaCount = 0
        etaIndices.clear()
        etaValues.clear()
    }

    private fun solvable(x: F64IndexedVector): F64SparseFactorization {
        requireShape(x.size == n) { "solve: x size ${x.size} != $n" }
        val factors = base ?: throw SingularMatrix(SINGULAR_POSITION_UNKNOWN, "solve: no basis has been factorized")
        if (factors.singular) throw SingularMatrix(factors.failedAt, "solve: the basis is singular")
        return factors
    }

    override fun close() {
        if (closed) return
        closed = true
        base?.close()
    }

    private fun checkOpen() {
        check(!closed) { "product-form basis solver is closed" }
    }
}

/**
 * Above this the dense sweep wins: tracking which positions a solve reaches costs a depth-first pass over
 * the factor's column graph, which only pays while the reachable set stays well under `m`.
 */
private const val REACHABLE_FTRAN_MAX_DENSITY = 0.1
