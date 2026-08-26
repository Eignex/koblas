@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.host.F64SparseBlasAdapter
import kotlinx.cinterop.*

/** Sparse-times-dense products backed by CHOLMOD's `cholmod_sdmult`. */
public open class CholmodSparseBlas(
    /** Policy for this backend instance. */
    public val config: CholmodConfig = CholmodConfig(),
    level2Min: Int? = null,
) : F64SparseBlasAdapter(level2Min) {
    private val loader = CholmodLoader(config)
    private val common by lazy { if (loader.available) loader.common() else null }

    override val name: String get() = BackendNames.CHOLMOD
    override val priority: Int get() = HOST_BACKEND_PRIORITY
    final override val nativeAvailable: Boolean get() = common != null

    /** Why CHOLMOD is unusable, or null when it is usable. For diagnostics, not control flow. */
    public val unavailableReason: String?
        get() = if (nativeAvailable) {
            null
        } else {
            "libcholmod could not be opened or its sparse BLAS symbols did not resolve"
        }

    @Suppress("LongParameterList")
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
        val multiplied = multiply(alpha, a, transposeA, b, beta, c)
        if (!multiplied) portable.gemm(alpha, a, transposeA, b, transposeB = false, beta = beta, c = c)
    }

    private fun multiply(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
    ): Boolean {
        val functions = loader.functions ?: return false
        val shared = common ?: return false
        val sparse = describeGeneral(a)
        try {
            return memScoped {
                val alphaPair = allocArray<DoubleVar>(2).also {
                    it[0] = alpha
                    it[1] = 0.0
                }
                val betaPair = allocArray<DoubleVar>(2).also {
                    it[0] = beta
                    it[1] = 0.0
                }
                val x = allocArray<DoubleVar>(maxOf(b.data.size, 1))
                for (index in b.data.indices) x[index] = b.data[index]
                val y = allocArray<DoubleVar>(maxOf(c.data.size, 1))
                for (index in c.data.indices) y[index] = c.data[index]
                val xDense = dense(b.rows, b.cols, x)
                val yDense = dense(c.rows, c.cols, y)
                val flag = if (transposeA) CHOLMOD_TRUE else 0
                val ok = functions.sdmult(sparse, flag, alphaPair, betaPair, xDense, yDense, shared)
                if (ok != CHOLMOD_TRUE) return@memScoped false
                for (index in c.data.indices) c.data[index] = y[index]
                true
            }
        } finally {
            freeMatrix(sparse)
        }
    }

    private fun MemScope.dense(rows: Int, columns: Int, values: CPointer<DoubleVar>): CPointer<ByteVar> =
        allocArray<ByteVar>(CHOLMOD_DENSE_BYTES).also { dense ->
            sizeAt(dense, CHOLMOD_DENSE_NROW, rows.toLong())
            sizeAt(dense, CHOLMOD_DENSE_NCOL, columns.toLong())
            sizeAt(dense, CHOLMOD_DENSE_NZMAX, rows.toLong() * columns)
            sizeAt(dense, CHOLMOD_DENSE_D, rows.toLong())
            pointerAt(dense, CHOLMOD_DENSE_X, values)
            pointerAt(dense, CHOLMOD_DENSE_Z, null)
            intAt(dense, CHOLMOD_DENSE_XTYPE, CHOLMOD_REAL)
            intAt(dense, CHOLMOD_DENSE_DTYPE, CHOLMOD_DOUBLE)
        }
}
