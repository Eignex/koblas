@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.host.F64SparseBlasAdapter
import kotlinx.cinterop.*

/**
 * The sparse matrix products backed by CHOLMOD's `cholmod_sdmult`.
 *
 * A backend of its own, unlike the Cholesky, which is a routine the SuiteSparse factorization bindings carry.
 * Nothing else fills the sparse BLAS half natively, so there is no library here for this one to take it from
 * and no reason to hide it behind another.
 *
 * `cholmod_sdmult` is `Y = alpha·op(A)·X + beta·Y` for a dense `X` of any column count, and the matrix-matrix
 * product is all that goes through it: the call costs the same whatever the column count, so a single
 * right-hand side has nothing to amortise it over. The adapter keeps `gemv` and the sparse-times-sparse
 * product portable for that reason, and this fills what is left.
 *
 * The operands are read where they lie, pinned for the length of the call, so nothing is copied to reach the
 * library. That is what a matrix answering in one call buys over one spanning an analysis and a numeric pass,
 * which is why the factorization bindings beside this copy theirs.
 */
public open class CholmodSparseBlas(
    /** Policy for this backend instance. */
    public val config: CholmodConfig = CholmodConfig(),
    level2Min: Int? = null,
) : F64SparseBlasAdapter(level2Min) {
    private val loader = CholmodLoader(config)

    /**
     * One `cholmod_common` for every product this backend answers, since starting one is what CHOLMOD
     * charges for its workspace. Nothing outlives the process, so it is never finished.
     */
    private val common: CPointer<ByteVar> by lazy { loader.common() }

    override val name: String get() = BackendNames.CHOLMOD
    override val priority: Int get() = HOST_BACKEND_PRIORITY
    final override val nativeAvailable: Boolean get() = loader.available

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
        // Every operand is read through a pinned address, and an empty array has none to take.
        val empty = a.nnz == 0 || b.data.isEmpty() || c.data.isEmpty()
        val multiplied = !empty && multiply(alpha, a, transposeA, b, beta, c)
        if (!multiplied) portable.gemm(alpha, a, transposeA, b, transposeB = false, beta = beta, c = c)
    }

    /**
     * `C = alpha · op(A) · B + beta · C` through `cholmod_sdmult`, or false when it refused the operands.
     *
     * It validates its arguments before it writes anything, so a refusal leaves [c] as it was and the caller
     * can still answer portably.
     */
    @Suppress("LongParameterList") // the BLAS dgemm signature less the flags this one does not take
    private fun multiply(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
    ): Boolean {
        val functions = loader.functions ?: return false
        return pinned(a.colPtr, a.rowIdx, a.values, b.data, c.data) { colPtr, rowIdx, values, x, y ->
            memScoped {
                functions.sdmult(
                    describeGeneral(a, colPtr, rowIdx, values),
                    if (transposeA) CHOLMOD_TRUE else 0,
                    scalarPair(alpha),
                    scalarPair(beta),
                    describeDense(b.rows, b.cols, x),
                    describeDense(c.rows, c.cols, y),
                    common,
                ) == CHOLMOD_TRUE
            }
        }
    }
}

/**
 * Runs [body] with the five operand arrays pinned, each as the address of its first entry.
 *
 * Written out rather than nested at the call site, where five `usePinned` blocks would bury the one call
 * they exist for.
 */
@Suppress("LongParameterList") // one parameter per operand array a sparse-times-dense product reads
private inline fun <R> pinned(
    colPtr: IntArray,
    rowIdx: IntArray,
    values: DoubleArray,
    x: DoubleArray,
    y: DoubleArray,
    body: (
        colPtr: CPointer<IntVar>,
        rowIdx: CPointer<IntVar>,
        values: CPointer<DoubleVar>,
        x: CPointer<DoubleVar>,
        y: CPointer<DoubleVar>,
    ) -> R,
): R = colPtr.usePinned { p ->
    rowIdx.usePinned { i ->
        values.usePinned { v ->
            x.usePinned { xs ->
                y.usePinned { ys ->
                    body(p.addressOf(0), i.addressOf(0), v.addressOf(0), xs.addressOf(0), ys.addressOf(0))
                }
            }
        }
    }
}

/** [value] as the complex pair CHOLMOD takes a scalar as, of which a real matrix reads only the first half. */
private fun NativePlacement.scalarPair(value: Double): CPointer<DoubleVar> = allocArray<DoubleVar>(2).also {
    it[0] = value
    it[1] = 0.0
}
