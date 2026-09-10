package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.dense.scaleTriangle
import com.eignex.koblas.sparse.internal.SparseAccumulationKernels
import com.eignex.koblas.sparse.internal.multiplyFromTheLeft
import com.eignex.koblas.sparse.internal.multiplyFromTheRight
import com.eignex.koblas.sparse.internal.multiplySparse
import com.eignex.koblas.sparse.internal.multiplySparseInto
import com.eignex.koblas.sparse.internal.stableFor
import com.eignex.koblas.sparse.internal.symmetricRankInto
import com.eignex.koblas.sparse.internal.symmetricRankProduct
import com.eignex.koblas.sparse.internal.transposeCsc
import com.eignex.koblas.sparse.internal.trmmLeftCore
import com.eignex.koblas.sparse.internal.trmmRightCore
import com.eignex.koblas.sparse.internal.trmvCore
import com.eignex.koblas.sparse.internal.trsmLeftCore
import com.eignex.koblas.sparse.internal.trsmRightCore
import com.eignex.koblas.sparse.internal.trsvCore
import com.eignex.koblas.sparse.internal.withExplicitDiagonal
import com.eignex.koblas.sparse.internal.withStableDense
import com.eignex.koblas.sparse.internal.withStableSparse
import com.eignex.koblas.sparse.internal.withSymmetricRankScratch

/** Dense right-hand sides processed per walk of portable CSC storage. */
internal const val REFERENCE_SPARSE_RHS_WIDTH: Int = 4

/**
 * Shared sparse matrix algorithms bound to one immutable dense-kernel implementation.
 *
 * Dense vector kernels are used by sparse matrix scaling and dense right-hand sides.
 */
@Suppress("TooManyFunctions") // the sparse BLAS surface
internal class SparseAlgorithms(
    private val vectorKernels: DenseVectorKernels,
    private val indexedKernels: IndexedSparseKernels,
    private val sparsePanelKernels: SparsePanelKernels,
) : SparseBlas {
    override fun prepare(a: SparseMatrix): PreparedSparseMatrix = PreparedSparseMatrix(a, this)

    @Suppress("LongParameterList") // the BLAS dgemv signature
    override fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        requireGemvShape(a, transpose, x.size, y.size)
        if (a.rows == 0 || a.cols == 0) return
        applyBeta(vectorKernels, y, 0, y.size, beta)
        if (alpha == 0.0) return
        if (transpose) {
            for (j in 0 until a.cols) {
                val sum = indexedKernels.dotDense(a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], x)
                y[j] += alpha * sum
            }
        } else {
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                indexedKernels.axpy(a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], xj, y)
            }
        }
    }

    override fun transpose(a: SparseMatrix): SparseMatrix = transposeCsc(a)

    override fun symv(alpha: Double, a: SparseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireSquare(a, "symv")
        requireShape(x.size == a.rows) { "symv: x length ${x.size} != ${a.rows}" }
        requireShape(y.size == a.rows) { "symv: y length ${y.size} != ${a.rows}" }
        if (alpha == 0.0) {
            applyBeta(vectorKernels, y, 0, y.size, beta)
            return
        }
        val stableA = a.stableFor(y)
        val stableX = if (x === y) x.copyOf() else x
        applyBeta(vectorKernels, y, 0, y.size, beta)
        for (column in 0 until stableA.cols) {
            sparsePanelKernels.symmetricVectorColumn(
                alpha,
                column,
                stableA.rowIdx,
                stableA.values,
                stableA.colPtr[column],
                stableA.colPtr[column + 1],
                stableX,
                y,
                lower,
            )
        }
    }

    @Suppress("LongParameterList")
    override fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSquare(a, "symm")
        requireShape(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        if (right) {
            requireShape(b.cols == a.rows) { "symm right: B has ${b.cols} cols, expected ${a.rows}" }
        } else {
            requireShape(b.rows == a.rows) { "symm: B has ${b.rows} rows, expected ${a.rows}" }
        }
        if (alpha == 0.0) {
            applyBeta(vectorKernels, c.data, 0, c.data.size, beta)
            return
        }
        withStableSparse(a, c.data, workspace) { stableA ->
            withStableDense(b, c.data, workspace) { stableB ->
                applyBeta(vectorKernels, c.data, 0, c.data.size, beta)
                for (column in 0 until stableA.cols) {
                    if (right) {
                        sparsePanelKernels.symmetricRightColumn(
                            alpha,
                            column,
                            stableA.rowIdx,
                            stableA.values,
                            stableA.colPtr[column],
                            stableA.colPtr[column + 1],
                            stableB.data,
                            c.data,
                            stableB.rows,
                            lower,
                        )
                    } else {
                        sparsePanelKernels.symmetricLeftColumn(
                            alpha,
                            column,
                            stableA.rowIdx,
                            stableA.values,
                            stableA.colPtr[column],
                            stableA.colPtr[column + 1],
                            stableB.data,
                            c.data,
                            stableB.rows,
                            stableB.cols,
                            lower,
                        )
                    }
                }
            }
        }
    }

    override fun trsv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trsv")
        val n = a.rows
        requireShape(x.size == n) { "trsv: x length ${x.size} != $n" }
        trsvCore(sparsePanelKernels, a, x, lower, transpose, unitDiag)
    }

    override fun trmv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trmv")
        val n = a.rows
        requireShape(x.size == n) { "trmv: x length ${x.size} != $n" }
        trmvCore(sparsePanelKernels, a.stableFor(x), x, lower, transpose, unitDiag)
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature, plus the side the sparse operand sits on
    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        right: Boolean,
        workspace: Workspace?,
    ) {
        // Multiplying the dense operand by the sparse one from the right is this product with the operands
        // the other way round, so the same derivation answers both.
        val (m, k, n) = if (right) {
            requireGemmShape(b, transposeB, a, transposeA, c)
        } else {
            requireGemmShape(a, transposeA, b, transposeB, c)
        }
        if (alpha == 0.0) {
            applyBeta(vectorKernels, c.data, 0, c.data.size, beta)
            return
        }
        withStableSparse(a, c.data, workspace) { stableA ->
            withStableDense(b, c.data, workspace) { stableB ->
                applyBeta(vectorKernels, c.data, 0, c.data.size, beta)
                if (right) {
                    multiplyFromTheRight(
                        sparsePanelKernels,
                        alpha,
                        stableA,
                        transposeA,
                        stableB,
                        transposeB,
                        c,
                        m,
                        workspace,
                    )
                } else {
                    multiplyFromTheLeft(
                        sparsePanelKernels,
                        alpha,
                        stableA,
                        transposeA,
                        stableB,
                        transposeB,
                        c,
                        m,
                        n,
                        k,
                        workspace,
                    )
                }
            }
        }
    }

    /** `C += alpha · op(A) · op(B)`, reusing each walk of the sparse operand over a small RHS panel. */

    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
    ): SparseMatrix {
        val aRows = if (transposeA) a.cols else a.rows
        val aCols = if (transposeA) a.rows else a.cols
        val bRows = if (transposeB) b.cols else b.rows
        val bCols = if (transposeB) b.rows else b.cols
        requireShape(aCols == bRows) { "gemm: op(A) is ${aRows}x$aCols but op(B) is ${bRows}x$bCols" }
        val left = oriented(a, transposeA, alpha != 0.0)
        val right = oriented(b, transposeB, alpha != 0.0)
        return multiplySparse(left, right, alpha)
    }

    @Suppress("LongParameterList")
    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        requireGemmShape(a, transposeA, b, transposeB, c)
        if (alpha == 0.0) {
            applyBeta(vectorKernels, c.data, 0, c.data.size, beta)
            return
        }
        withStableSparse(a, c.data, workspace) { stableA ->
            withStableSparse(b, c.data, workspace) { stableB ->
                val left = oriented(stableA, transposeA, true)
                val right = oriented(stableB, transposeB, true)
                applyBeta(vectorKernels, c.data, 0, c.data.size, beta)
                multiplySparseInto(alpha, left, right, c)
            }
        }
    }

    @Suppress("LongParameterList")
    override fun syrk(
        alpha: Double,
        a: SparseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        if (alpha == 0.0) {
            scaleTriangle(vectorKernels, c.data, n, beta, lower)
            return
        }
        withStableSparse(a, c.data, workspace) { stableA ->
            scaleTriangle(vectorKernels, c.data, n, beta, lower)
            withSymmetricRankScratch(workspace, n, stableA.rows, stableA.nnz) {
                    sums,
                    touchedAt,
                    touched,
                    rowPointers,
                    adjacentColumns,
                    adjacentPositions,
                    rowCursor,
                ->
                symmetricRankInto(
                    alpha, stableA, transpose, c, lower, sums, touchedAt, touched,
                    rowPointers, adjacentColumns, adjacentPositions, rowCursor,
                )
            }
        }
    }

    override fun syrk(a: SparseMatrix, transpose: Boolean, lower: Boolean): SparseMatrix =
        symmetricRankProduct(a, transpose, lower)

    override fun addScaled(alpha: Double, a: SparseMatrix, transposeA: Boolean, b: SparseMatrix): SparseMatrix {
        val rows = if (transposeA) a.cols else a.rows
        val cols = if (transposeA) a.rows else a.cols
        requireShape(rows == b.rows && cols == b.cols) {
            "addScaled: op(A) is ${rows}x$cols but B is ${b.rows}x${b.cols}"
        }
        val left = oriented(a, transposeA, alpha != 0.0)
        val pointers = IntArray(left.cols + 1)
        val rowIndices = IntArray(left.nnz + b.nnz)
        val values = DoubleArray(rowIndices.size)
        var count = 0
        for (column in 0 until left.cols) {
            count += SparseAccumulationKernels.mergeScaledColumns(
                alpha,
                left.rowIdx,
                left.values,
                left.colPtr[column],
                left.colPtr[column + 1],
                b.rowIdx,
                b.values,
                b.colPtr[column],
                b.colPtr[column + 1],
                rowIndices,
                values,
                count,
            )
            pointers[column + 1] = count
        }
        return SparseMatrix.wrapTrusted(
            left.rows,
            left.cols,
            pointers,
            rowIndices.copyOf(count),
            values.copyOf(count),
        )
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: SparseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        val n = requireTriangularMatrixShape(a, b, right, "trsm")
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        if (n == 0) return
        if (alpha != 1.0) vectorKernels.scale(b.data, 0, alpha, b.data.size)
        val rightHandSides = if (right) b.rows else b.cols
        if (rightHandSides == 0) return
        if (right) {
            withExplicitDiagonal(a, n, unitDiag, workspace) { diagonal ->
                trsmRightCore(sparsePanelKernels, a, b, lower, !transpose, diagonal)
            }
        } else {
            workspace.borrow(REFERENCE_SPARSE_RHS_WIDTH) { work ->
                withExplicitDiagonal(a, n, unitDiag, workspace) { diagonal ->
                    trsmLeftCore(sparsePanelKernels, a, b, lower, transpose, diagonal, work)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: SparseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) {
        val n = requireTriangularMatrixShape(a, b, right, "trmm")
        if (alpha == 0.0) {
            b.data.fill(0.0)
            return
        }
        val triangle = a.stableFor(b.data)
        if (alpha != 1.0) vectorKernels.scale(b.data, 0, alpha, b.data.size)
        // Read once for every right-hand side rather than once per trmvCore call, as trsm does.
        val diagonal = if (unitDiag) null else DoubleArray(n) { triangle[it, it] }
        if (right) {
            trmmRightCore(sparsePanelKernels, triangle, b, lower, transpose, unitDiag, diagonal)
        } else {
            trmmLeftCore(sparsePanelKernels, triangle, b, lower, transpose, unitDiag, diagonal)
        }
    }

    private fun oriented(a: SparseMatrix, transpose: Boolean, readValues: Boolean): SparseMatrix {
        if (!transpose) return a
        if (readValues) return transposeCsc(a)
        val patternOnly = SparseMatrix.wrapTrusted(
            a.rows,
            a.cols,
            a.copyColumnPointers(),
            a.copyRowIndices(),
            DoubleArray(a.nnz),
        )
        return transposeCsc(patternOnly)
    }
}
