@file:OptIn(ExperimentalForeignApi::class, ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.noManagedAllocation
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseFactorizationReport
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_BYTES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_D
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_DTYPE
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_NCOL
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_NROW
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_NZMAX
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_X
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_XTYPE
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_Z
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DOUBLE
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_REAL
import com.eignex.koblas.sparse.host.cholmod.intAt
import com.eignex.koblas.sparse.host.cholmod.pointerAt
import com.eignex.koblas.sparse.host.cholmod.sizeAt
import com.eignex.koblas.sparse.requireLeastSquaresShapes
import kotlinx.cinterop.*
import kotlin.experimental.ExperimentalNativeApi
import kotlin.native.concurrent.ThreadLocal
import kotlin.native.ref.createCleaner

/** A SPQR factorization with deterministic close and cleaner fallback for its native factor. */
public class SpqrQrFactorization internal constructor(
    private val handle: SpqrHandle,
    private val functions: SpqrFunctions,
    private val explicitFactors: () -> SpqrExplicit,
    override val m: Int,
    override val n: Int,
    private val ordering: String,
) : F64SparseQrFactorization {

    internal class SpqrHandle(
        val factor: COpaquePointer,
        val common: CPointer<ByteVar>,
        private val functions: SpqrFunctions,
        rows: Int,
    ) {
        val rhs: CPointer<DoubleVar> = nativeHeap.allocArray(maxOf(rows, 1))
        val dense: CPointer<ByteVar> = nativeHeap.allocArray(CHOLMOD_DENSE_BYTES)
        val slot: COpaquePointerVar = nativeHeap.alloc()

        init {
            sizeAt(dense, CHOLMOD_DENSE_NROW, rows.toLong())
            sizeAt(dense, CHOLMOD_DENSE_NCOL, 1L)
            sizeAt(dense, CHOLMOD_DENSE_NZMAX, rows.toLong())
            sizeAt(dense, CHOLMOD_DENSE_D, rows.toLong())
            pointerAt(dense, CHOLMOD_DENSE_X, rhs)
            pointerAt(dense, CHOLMOD_DENSE_Z, null)
            intAt(dense, CHOLMOD_DENSE_XTYPE, CHOLMOD_REAL)
            intAt(dense, CHOLMOD_DENSE_DTYPE, CHOLMOD_DOUBLE)
        }

        fun release() {
            slot.value = factor
            functions.freeQr(slot.ptr, common)
            nativeHeap.free(slot)
            nativeHeap.free(dense)
            nativeHeap.free(rhs)
            nativeHeap.free(common)
        }
    }

    private val lifecycle = NativeResourceLifecycle("SPQR factorization", handle::release)

    @Suppress("unused") // the cleaner runs when this property becomes unreachable, which is the point
    private val cleaner = createCleaner(lifecycle) { it.close() }

    /**
     * `R`, its ordering, and the rank, from a second factorization: SPQR's C interface reads neither factor
     * out of the object its solves use, and that second one chooses its own ordering, so `Q·R = A·P` across
     * the two does not hold.
     */
    private val explicit: SpqrExplicit by lazy { explicitFactors() }

    private val factors: SpqrExplicit get() = lifecycle.withResource { explicit }

    override val rank: Int get() = factors.rank

    override val nnz: Int get() = factors.r.nnz

    override val r: F64SparseMatrix get() = factors.r

    override val columnOrder: IntArray get() = factors.columnOrder.copyOf()

    override fun solveAllocation(): AllocationCapability = noManagedAllocation

    override fun applyQInto(y: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireShape(y.size == m) { "applyQ: y size ${y.size}, expected $m" }
        requireShape(out.size == m) { "applyQ: out size ${out.size}, expected $m" }
        anchoring {
            for (i in 0 until m) handle.rhs[i] = y[i]
            val method = if (transpose) SPQR_QTX else SPQR_QX
            val applied = functions.qmult(method, handle.factor.reinterpret(), handle.dense, handle.common)
            check(applied != null) { "SuiteSparseQR_C_qmult failed on a factorization it produced" }
            try {
                val values = pointerAt(applied.reinterpret(), CHOLMOD_DENSE_X)!!.reinterpret<DoubleVar>()
                for (i in 0 until m) out[i] = values[i]
            } finally {
                handle.slot.value = applied
                functions.freeDense(handle.slot.ptr, handle.common)
            }
        }
        return out
    }

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

    override fun solveInto(b: DoubleArray, out: DoubleArray, workspace: Workspace?): DoubleArray {
        requireLeastSquaresShapes(m, n, b, out)
        anchoring {
            for (i in 0 until m) handle.rhs[i] = b[i]
            val projected = functions.qmult(SPQR_QTX, handle.factor.reinterpret(), handle.dense, handle.common)
            check(projected != null) { "SuiteSparseQR_C_qmult failed on a factorization it produced" }
            try {
                val solved = functions.solve(
                    SPQR_RETX_EQUALS_B,
                    handle.factor.reinterpret(),
                    projected.reinterpret(),
                    handle.common,
                )
                check(solved != null) { "SuiteSparseQR_C_solve failed on a factorization it produced" }
                try {
                    val values = pointerAt(solved.reinterpret(), CHOLMOD_DENSE_X)!!.reinterpret<DoubleVar>()
                    for (i in 0 until n) out[i] = values[i]
                } finally {
                    handle.slot.value = solved
                    functions.freeDense(handle.slot.ptr, handle.common)
                }
            } finally {
                handle.slot.value = projected
                functions.freeDense(handle.slot.ptr, handle.common)
            }
        }
        return out
    }

    override fun close(): Unit = lifecycle.close()

    private fun <R> anchoring(body: () -> R): R = lifecycle.withResource {
        val previous = AnchoredSpqr.held
        AnchoredSpqr.held = this
        try {
            body()
        } finally {
            AnchoredSpqr.held = previous
        }
    }
}

@ThreadLocal
private object AnchoredSpqr {
    var held: SpqrQrFactorization? = null
}
