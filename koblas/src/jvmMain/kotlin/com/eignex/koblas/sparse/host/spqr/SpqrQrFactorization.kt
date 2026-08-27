package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.noManagedAllocation
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseFactorizationReport
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.host.cholmod.CholmodMatrix
import com.eignex.koblas.sparse.requireLeastSquaresShapes
import java.lang.ref.Reference

/** A SPQR factorization with deterministic close and cleaner fallback for its native factor. */
public class SpqrQrFactorization internal constructor(
    private val factor: SpqrFactor,
    private val calls: SpqrCalls,
    private val source: F64SparseMatrix,
    override val m: Int,
    override val n: Int,
    private val ordering: String,
) : F64SparseQrFactorization {
    private val lifecycle = NativeResourceLifecycle("SPQR factorization") { calls.free(factor) }
    private val cleanable = nativeCleaner.register(this, lifecycle)

    /**
     * `R`, its ordering, and the rank, from a second factorization: SPQR's C interface reads neither factor
     * out of the object its solves use, and that second one chooses its own ordering, so `Q·R = A·P` across
     * the two does not hold.
     */
    private val explicit: SpqrExplicit by lazy {
        CholmodMatrix.generalOf(source).use { operand ->
            checkNotNull(calls.explicitR(operand, n)) { "SuiteSparseQR_i_C_QR failed on a matrix it factorized" }
        }
    }

    private val factors: SpqrExplicit get() = lifecycle.withResource { explicit }

    override val rank: Int get() = factors.rank

    override val nnz: Int get() = factors.r.nnz

    override val r: F64SparseMatrix get() = factors.r

    override val columnOrder: IntArray get() = factors.columnOrder.copyOf()

    override fun applyQInto(y: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        lifecycle.withResource {
            requireShape(y.size == m) { "applyQ: y size ${y.size}, expected $m" }
            requireShape(out.size == m) { "applyQ: out size ${out.size}, expected $m" }
            try {
                val applied = checkNotNull(calls.applyQ(factor, if (transpose) SPQR_QTX else SPQR_QX, y, m)) {
                    "SuiteSparseQR_C_qmult failed on a factorization it produced"
                }
                applied.copyInto(out)
            } finally {
                Reference.reachabilityFence(this)
            }
            out
        }

    override fun solveAllocation(): AllocationCapability = noManagedAllocation

    override fun report(): F64SparseFactorizationReport = F64SparseFactorizationReport(
        provider = "spqr",
        order = n,
        factorNonzeros = nnz,
        // SPQR's statistics sit at trailing `cholmod_common` offsets that move between releases.
        reciprocalPivotRange = null,
        ordering = ordering,
        columnPermutation = factors.columnOrder.toList(),
        details = mapOf("rows" to m.toString(), "rank" to rank.toString()),
    )

    override fun solveInto(b: DoubleArray, out: DoubleArray, workspace: Workspace?): DoubleArray =
        lifecycle.withResource {
            requireLeastSquaresShapes(m, n, b, out)
            try {
                val solved = checkNotNull(calls.solveLeastSquares(factor, b, n)) {
                    "SuiteSparseQR_C_solve failed on a factorization it produced"
                }
                solved.copyInto(out)
            } finally {
                Reference.reachabilityFence(this)
            }
            out
        }

    override fun close(): Unit = cleanable.clean()
}
