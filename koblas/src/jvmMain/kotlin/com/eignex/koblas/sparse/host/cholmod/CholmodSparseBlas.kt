package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.host.F64SparseBlasAdapter

/**
 * The sparse matrix products backed by CHOLMOD's `cholmod_sdmult`.
 *
 * A backend of its own, unlike the Cholesky, which is a routine the SuiteSparse factorization bindings carry.
 * Nothing else fills the sparse BLAS half natively, so there is no library here for this one to take it from
 * and no reason to hide it behind another.
 *
 * `cholmod_sdmult` is `Y = alpha·op(A)·X + beta·Y` for a dense `X` of any column count. Only the matrix-matrix
 * product goes through it. The call costs the same whatever the column count, so one right-hand side has
 * nothing to amortise it over and measured slower than the portable loop at every size; the adapter keeps
 * `gemv` and the sparse-times-sparse product portable for that reason, and this fills what is left.
 */
public open class CholmodSparseBlas(
    /** Policy for this backend instance. */
    public val config: CholmodConfig = CholmodConfig(),
    level2Min: Int? = null,
) : F64SparseBlasAdapter(level2Min) {
    private val calls = CholmodCalls(config)

    override val name: String get() = BackendNames.CHOLMOD
    override val priority: Int get() = HOST_BACKEND_PRIORITY
    final override val nativeAvailable: Boolean get() = calls.available

    /** Why CHOLMOD is unusable, or null when it is usable. For diagnostics, not control flow. */
    public val unavailableReason: String? get() = calls.unavailableReason

    @Suppress("LongParameterList") // the BLAS dgemm signature less the flags this one does not take
    final override fun gemmNative(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        requireShape(k == b.rows) { "gemm: op(A) is ${m}x$k but B has ${b.rows} rows" }
        requireShape(c.rows == m && c.cols == b.cols) {
            "gemm: C is ${c.rows}x${c.cols}, expected ${m}x${b.cols}"
        }
        val multiplied = CholmodMatrix.generalOf(a).use { matrix ->
            calls.sdmult(
                matrix,
                transposeA,
                alpha,
                b.data,
                beta,
                c.data,
                columns = b.cols,
                xRows = b.rows,
                yRows = c.rows,
            )
        }
        if (!multiplied) portable.gemm(alpha, a, transposeA, b, transposeB = false, beta = beta, c = c)
    }
}
