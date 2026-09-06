package com.eignex.koblas.sparse.host

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseLuFactorization

/**
 * A native factorization of `E·A` presented as one of `A`.
 *
 * The library was handed row-scaled values, so a forward solve scales its right-hand side going in and a
 * transposed one scales its result coming out, by the reasoning [applyF64Equilibration] records. Everything
 * else is the library's own answer about the factors it holds.
 *
 * Written out rather than delegated with `by`: [F64SparseLuFactorization] gives its dense solves and its
 * `solve` overloads default bodies that call back into [solveInto], and a delegating class would send those
 * to the wrapped factorization, which would answer them without undoing the scaling.
 */
internal class EquilibratedSparseLu(private val inner: F64SparseLuFactorization, private val scale: DoubleArray) :
    F64SparseLuFactorization {
    override val n: Int get() = inner.n
    override val failedAt: Int get() = inner.failedAt
    override val nnz: Int get() = inner.nnz
    override val rcond: Double get() = inner.rcond

    override val l: F64SparseMatrix get() = inner.l
    override val u: F64SparseMatrix get() = inner.u
    override val rowOrder: IntArray get() = inner.rowOrder
    override val columnOrder: IntArray get() = inner.columnOrder
    override val offDiagonal: F64SparseMatrix get() = inner.offDiagonal

    /** The factors are of `E·A`, and these are that `E`. */
    override val rowScaling: DoubleArray get() = scale.copyOf()

    /**
     * A forward solve writes the scaled right-hand side into the destination and solves it in place, so the
     * wrapped factorization always sees an aliasing solve however the caller asked for it.
     */
    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        inner.solveAllocation(aliasing = aliasing || !transpose, transpose = transpose)

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        // The destination carries the scaled right-hand side in, which it may since the solve overwrites it
        // and reads each position before writing it where the two alias.
        if (!transpose) {
            for (i in b.indices) out[i] = b[i] * scale[i]
            inner.solveInto(out, out, transpose = false, workspace = workspace)
            return out
        }
        inner.solveInto(b, out, transpose = true, workspace = workspace)
        applyF64Equilibration(out, scale)
        return out
    }

    override fun close(): Unit = inner.close()
}
