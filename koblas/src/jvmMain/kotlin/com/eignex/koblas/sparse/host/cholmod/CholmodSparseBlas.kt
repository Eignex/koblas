package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64PreparedSparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.host.F64SparseBlasAdapter
import com.eignex.koblas.sparse.sparseSnapshotOf

/**
 * The sparse matrix products backed by CHOLMOD's `cholmod_sdmult`.
 *
 * A backend of its own, unlike the Cholesky, which is a routine the SuiteSparse factorization bindings carry.
 * Nothing else fills the sparse BLAS half natively, so there is no library here for this one to take it from
 * and no reason to hide it behind another.
 *
 * `cholmod_sdmult` is `Y = alpha·op(A)·X + beta·Y` for a dense `X` of any column count.
 */
public open class CholmodSparseBlas(
    /** Policy for this backend instance. */
    public val config: CholmodConfig = CholmodConfig(),
) : F64SparseBlasAdapter() {
    private val calls = CholmodCalls(config)

    override val name: String get() = BackendNames.CHOLMOD
    override val priority: Int get() = HOST_BACKEND_PRIORITY
    final override val nativeAvailable: Boolean get() = calls.available

    /** Why CHOLMOD is unusable, or null when it is usable. For diagnostics, not control flow. */
    public val unavailableReason: String? get() = calls.unavailableReason

    final override fun route(query: F64RouteQuery): BackendRoute? {
        if (query !is F64RouteQuery.PreparedSparseProduct) return super.route(query)
        return nativeRoute(query, this, portable.name)
    }

    /** Copies [a] once into a caller-owned CHOLMOD descriptor for repeated products. */
    final override fun prepare(a: F64SparseMatrix): F64PreparedSparseMatrix {
        if (!nativeAvailable) return super.prepare(a)
        val snapshot = sparseSnapshotOf(a)
        return CholmodPreparedSparseMatrix(
            snapshot,
            CholmodMatrix.retainedGeneralOf(snapshot),
            calls,
        )
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature less the flags this one does not take
    final override fun gemmNative(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        workspace: Workspace?,
    ) {
        requireGemmShape(a.rows, a.cols, transposeA, b, transposeB = false, c = c)
        // Built and closed inside this call, so nothing here outlives it: no snapshot of the caller's
        // arrays, which cannot change under a descriptor with this lifetime; a confined arena rather than
        // the shared one prepare needs, whose close costs a thread handshake; and no cleaner registration
        // for an object that cannot escape. Going through prepare paid all three on every product.
        CholmodMatrix.generalOf(a).use { matrix ->
            if (!calls.sdmult(matrix, transposeA, alpha, b.data, beta, c.data, b.cols, b.rows, c.rows)) {
                F64ReferenceSparseLinearAlgebra.gemm(alpha, a, transposeA, b, false, beta, c, workspace = workspace)
            }
        }
    }
}

private class CholmodPreparedSparseMatrix(
    private val snapshot: F64SparseMatrix,
    private val matrix: CholmodMatrix,
    private val calls: CholmodCalls,
) : F64PreparedSparseMatrix {
    private val lifecycle = NativeResourceLifecycle("prepared CHOLMOD matrix", matrix::close)
    private val cleanable = nativeCleaner.register(this, lifecycle)

    override val rows: Int get() = snapshot.rows
    override val cols: Int get() = snapshot.cols
    override val nnz: Int get() = snapshot.nnz

    override fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        val shape = requireGemvShape(rows, cols, transpose, x.size, y.size)
        lifecycle.withResource {
            if (!calls.sdmult(matrix, transpose, alpha, x, beta, y, 1, shape.inputs, shape.outputs)) {
                F64ReferenceSparseLinearAlgebra.gemv(alpha, snapshot, x, beta, y, transpose)
            }
        }
    }

    override fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        workspace: Workspace?,
    ) {
        requireGemmShape(rows, cols, transposeA, b, transposeB = false, c = c)
        lifecycle.withResource {
            if (!calls.sdmult(matrix, transposeA, alpha, b.data, beta, c.data, b.cols, b.rows, c.rows)) {
                F64ReferenceSparseLinearAlgebra.gemm(
                    alpha,
                    snapshot,
                    transposeA,
                    b,
                    false,
                    beta,
                    c,
                    workspace = workspace,
                )
            }
        }
    }

    override fun gemm(b: F64SparseMatrix): F64SparseMatrix = lifecycle.withResource {
        CholmodMatrix.generalOf(b).use { right ->
            calls.ssmult(matrix, right) ?: F64ReferenceSparseLinearAlgebra.gemm(snapshot, b)
        }
    }

    override fun close(): Unit = cleanable.clean()
}
