@file:OptIn(ExperimentalForeignApi::class, kotlin.experimental.ExperimentalNativeApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.sparse.F64PreparedSparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.host.F64SparseBlasAdapter
import com.eignex.koblas.sparse.sparseSnapshotOf
import kotlinx.cinterop.*
import kotlin.native.ref.createCleaner

/** Sparse-times-dense products backed by CHOLMOD's `cholmod_sdmult`. */
public open class CholmodSparseBlas(
    /** Policy for this backend instance. */
    public val config: CholmodConfig = CholmodConfig(),
) : F64SparseBlasAdapter() {
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

    final override fun route(query: F64RouteQuery): BackendRoute? {
        if (query !is F64RouteQuery.PreparedSparseProduct) return super.route(query)
        return nativeRoute(query, this, portable.name)
    }

    /** Copies [a] once into a caller-owned CHOLMOD descriptor for repeated products. */
    final override fun prepare(a: F64SparseMatrix): F64PreparedSparseMatrix {
        val functions = loader.functions ?: return super.prepare(a)
        val shared = common ?: return super.prepare(a)
        val snapshot = sparseSnapshotOf(a)
        return CholmodPreparedSparseMatrix(
            snapshot,
            describeGeneral(snapshot),
            functions,
            shared,
            nativeGemv = true,
            nativeGemm = true,
            nativeSparseProduct = true,
        )
    }

    @Suppress("LongParameterList")
    final override fun gemmNative(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        workspace: Workspace?,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        requireShape(k == b.rows) { "gemm: op(A) is ${m}x$k but B has ${b.rows} rows" }
        requireShape(c.rows == m && c.cols == b.cols) {
            "gemm: C is ${c.rows}x${c.cols}, expected ${m}x${b.cols}"
        }
        prepare(a).use { it.gemm(alpha, transposeA, b, beta, c, workspace) }
    }
}

private class CholmodPreparedSparseMatrix(
    private val snapshot: F64SparseMatrix,
    private val sparse: CPointer<ByteVar>,
    private val functions: CholmodFunctions,
    private val common: CPointer<ByteVar>,
    private val nativeGemv: Boolean,
    private val nativeGemm: Boolean,
    private val nativeSparseProduct: Boolean,
) : F64PreparedSparseMatrix {
    private val lifecycle = NativeResourceLifecycle("prepared CHOLMOD matrix") { freeMatrix(sparse) }

    @Suppress("unused")
    private val cleaner = createCleaner(lifecycle) { it.close() }

    override val rows: Int get() = snapshot.rows
    override val cols: Int get() = snapshot.cols
    override val nnz: Int get() = snapshot.nnz

    override fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        val xRows = if (transpose) rows else cols
        val yRows = if (transpose) cols else rows
        requireShape(x.size == xRows) { "gemv: x length ${x.size} != $xRows" }
        requireShape(y.size == yRows) { "gemv: y length ${y.size} != $yRows" }
        lifecycle.withResource {
            if (!nativeGemv || !multiply(alpha, transpose, x, beta, y, 1, xRows, yRows)) {
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
        val m = if (transposeA) cols else rows
        val k = if (transposeA) rows else cols
        requireShape(k == b.rows) { "gemm: op(A) is ${m}x$k but B has ${b.rows} rows" }
        requireShape(c.rows == m && c.cols == b.cols) {
            "gemm: C is ${c.rows}x${c.cols}, expected ${m}x${b.cols}"
        }
        lifecycle.withResource {
            if (!nativeGemm || !multiply(alpha, transposeA, b.data, beta, c.data, b.cols, b.rows, c.rows)) {
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
        if (!nativeSparseProduct) {
            F64ReferenceSparseLinearAlgebra.gemm(snapshot, b)
        } else {
            val right = describeGeneral(b)
            try {
                sparseProduct(right) ?: F64ReferenceSparseLinearAlgebra.gemm(snapshot, b)
            } finally {
                freeMatrix(right)
            }
        }
    }

    override fun close(): Unit = lifecycle.close()

    @Suppress("LongParameterList")
    private fun multiply(
        alpha: Double,
        transpose: Boolean,
        xValues: DoubleArray,
        beta: Double,
        yValues: DoubleArray,
        columns: Int,
        xRows: Int,
        yRows: Int,
    ): Boolean = memScoped {
        val alphaPair = allocArray<DoubleVar>(2).also {
            it[0] = alpha
            it[1] = 0.0
        }
        val betaPair = allocArray<DoubleVar>(2).also {
            it[0] = beta
            it[1] = 0.0
        }
        val x = allocArray<DoubleVar>(maxOf(xValues.size, 1))
        for (index in xValues.indices) x[index] = xValues[index]
        val y = allocArray<DoubleVar>(maxOf(yValues.size, 1))
        for (index in yValues.indices) y[index] = yValues[index]
        val xDense = dense(xRows, columns, x)
        val yDense = dense(yRows, columns, y)
        val flag = if (transpose) CHOLMOD_TRUE else 0
        if (functions.sdmult(sparse, flag, alphaPair, betaPair, xDense, yDense, common) != CHOLMOD_TRUE) {
            return@memScoped false
        }
        for (index in yValues.indices) yValues[index] = y[index]
        true
    }

    private fun sparseProduct(right: CPointer<ByteVar>): F64SparseMatrix? {
        val product = functions.ssmult(
            sparse,
            right,
            CHOLMOD_STYPE_GENERAL,
            CHOLMOD_TRUE,
            CHOLMOD_TRUE,
            common,
        ) ?: return null
        try {
            val descriptor = product.reinterpret<ByteVar>()
            val rows = sizeAt(descriptor, CHOLMOD_SPARSE_NROW).toInt()
            val cols = sizeAt(descriptor, CHOLMOD_SPARSE_NCOL).toInt()
            val p = checkNotNull(pointerAt(descriptor, CHOLMOD_SPARSE_P)).reinterpret<IntVar>()
            val colPtr = IntArray(cols + 1) { p[it] }
            val nnz = colPtr[cols]
            val i = checkNotNull(pointerAt(descriptor, CHOLMOD_SPARSE_I)).reinterpret<IntVar>()
            val x = checkNotNull(pointerAt(descriptor, CHOLMOD_SPARSE_X)).reinterpret<DoubleVar>()
            return F64SparseMatrix.wrap(
                rows,
                cols,
                colPtr,
                IntArray(nnz) { i[it] },
                DoubleArray(nnz) { x[it] },
            )
        } finally {
            memScoped {
                val slot = alloc<COpaquePointerVar>()
                slot.value = product
                functions.freeSparse(slot.ptr, common)
            }
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
