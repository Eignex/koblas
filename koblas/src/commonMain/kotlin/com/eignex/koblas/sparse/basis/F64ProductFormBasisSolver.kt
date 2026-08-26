package com.eignex.koblas.sparse.basis

import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireInBounds
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.F64SparseFactorization

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

    // The chain in update order, the pivot held apart from the spike so applying one is a loop over the
    // off-pivot entries and a single divide.
    private val etaPivotRow = ArrayList<Int>()
    private val etaPivot = ArrayList<Double>()
    private val etaIndices = ArrayList<IntArray>()
    private val etaValues = ArrayList<DoubleArray>()

    private val dense = DoubleArray(n)
    private val workspace = Workspace().apply { reserve(n, count = 3) }

    override val updateCount: Int get() = etaPivotRow.size

    override val singular: Boolean get() = base.let { it == null || it.singular }

    override val nnz: Int
        get() {
            var total = base?.nnz ?: 0
            for (entries in etaIndices) total += entries.size + 1
            return total
        }

    override fun refactorize(basicIndex: IntArray): Boolean {
        requireShape(basicIndex.size == n) { "refactorize: basicIndex size ${basicIndex.size} != $n" }
        for (t in 0 until n) requireInBounds(basicIndex[t], a.cols)
        // Factorized before any of this is committed, so a backend that throws rather than answering
        // singular leaves the solver on the basis it already holds instead of on one it cannot solve.
        val factors = lu.factor(basisMatrix(basicIndex))
        basicIndex.copyInto(this.basicIndex)
        dropChain()
        base = factors
        return !factors.singular
    }

    /** [expectedDensity] is not read: there is one sweep here to choose from, for the reason the class says. */
    override fun ftran(x: F64IndexedVector, expectedDensity: Double) {
        val factors = solvable(x)
        x.gather(dense)
        factors.solveInto(dense, dense, transpose = false, workspace = workspace)
        for (j in etaPivotRow.indices) applyEta(j)
        x.scatter(dense)
    }

    /** [expectedDensity] is not read, as in [ftran]. */
    override fun btran(x: F64IndexedVector, expectedDensity: Double) {
        val factors = solvable(x)
        x.gather(dense)
        for (j in etaPivotRow.indices.reversed()) applyEtaTransposed(j)
        factors.solveInto(dense, dense, transpose = true, workspace = workspace)
        x.scatter(dense)
    }

    override fun update(
        pivotRow: Int,
        entering: Int,
        spike: F64IndexedVector,
        pivotEta: F64IndexedVector?,
    ): BasisUpdate {
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
        etaPivotRow.add(pivotRow)
        etaPivot.add(pivot)
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
    private fun basisMatrix(basicIndex: IntArray): F64SparseMatrix {
        val colPtr = IntArray(n + 1)
        for (t in 0 until n) {
            var entries = 0
            a.forEachInColumn(basicIndex[t]) { _, _ -> entries++ }
            colPtr[t + 1] = colPtr[t] + entries
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
        etaPivotRow.clear()
        etaPivot.clear()
        etaIndices.clear()
        etaValues.clear()
    }

    private fun solvable(x: F64IndexedVector): F64SparseFactorization {
        requireShape(x.size == n) { "solve: x size ${x.size} != $n" }
        val factors = base ?: throw SingularMatrix(SINGULAR_POSITION_UNKNOWN, "solve: no basis has been factorized")
        if (factors.singular) throw SingularMatrix(factors.failedAt, "solve: the basis is singular")
        return factors
    }
}
