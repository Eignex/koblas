package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.sparse.PreparedSparseMatrix
import com.eignex.koblas.sparse.SparseBlas

/** Portable prepared ownership: an immutable CSC snapshot plus one lazily cached transpose. */
internal class ReferencePreparedSparseMatrix(a: SparseMatrix, private val algorithms: SparseBlas) :
    PreparedSparseMatrix {
    private val snapshot = sparseSnapshotOf(a)
    private var transposedSnapshot: SparseMatrix? = null
    private var closed = false

    override val rows: Int get() = snapshot.rows
    override val cols: Int get() = snapshot.cols
    override val nnz: Int get() = snapshot.nnz

    override fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        checkOpen()
        algorithms.gemv(alpha, snapshot, x, beta, y, transpose)
    }

    override fun symv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        checkOpen()
        algorithms.symv(alpha, snapshot, x, beta, y, lower)
    }

    override fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        gemm(alpha, transposeA, b, false, beta, c, false, workspace)
    }

    override fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        right: Boolean,
        workspace: Workspace?,
    ) {
        checkOpen()
        algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB, beta, c, right, workspace)
    }

    override fun symm(
        alpha: Double,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        checkOpen()
        algorithms.symm(alpha, snapshot, b, beta, c, lower, right, workspace)
    }

    override fun gemm(alpha: Double, transposeA: Boolean, b: SparseMatrix, transposeB: Boolean): SparseMatrix {
        checkOpen()
        return algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB)
    }

    override fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        checkOpen()
        algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB, beta, c, workspace)
    }

    override fun close() {
        closed = true
    }

    private fun checkOpen() {
        check(!closed) { "prepared sparse matrix is closed" }
    }

    private fun preparedOrientation(transpose: Boolean): SparseMatrix {
        if (!transpose) return snapshot
        val existing = transposedSnapshot
        if (existing != null) return existing
        return algorithms.transpose(snapshot).also { transposedSnapshot = it }
    }
}

/** Copies validated CSC storage so expert mutation cannot stale a prepared handle. */
private fun sparseSnapshotOf(a: SparseMatrix): SparseMatrix = SparseMatrix.wrap(
    a.rows,
    a.cols,
    a.copyColumnPointers(),
    a.copyRowIndices(),
    a.values.copyOf(),
)
