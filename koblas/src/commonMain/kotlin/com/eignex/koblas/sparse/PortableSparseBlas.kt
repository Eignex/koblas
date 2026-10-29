@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.scaleRegion
import com.eignex.koblas.dense.stagedRegion
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.sparse.internal.SparseAccumulationKernels
import com.eignex.koblas.sparse.internal.multiplyFromTheLeft
import com.eignex.koblas.sparse.internal.multiplyFromTheRight
import com.eignex.koblas.sparse.internal.multiplySparse
import com.eignex.koblas.sparse.internal.multiplySparseInto
import com.eignex.koblas.sparse.internal.multiplySymmetricFromTheLeft
import com.eignex.koblas.sparse.internal.pointerLength
import com.eignex.koblas.sparse.internal.symmetricRankInto
import com.eignex.koblas.sparse.internal.symmetricRankProduct
import com.eignex.koblas.sparse.internal.transposeCsc
import com.eignex.koblas.sparse.internal.triangularLeftCore
import com.eignex.koblas.sparse.internal.trmmRightCore
import com.eignex.koblas.sparse.internal.trmvCore
import com.eignex.koblas.sparse.internal.trsmRightCore
import com.eignex.koblas.sparse.internal.trsvCore
import com.eignex.koblas.sparse.internal.withExplicitDiagonal
import com.eignex.koblas.sparse.internal.withSymmetricRankScratch

/**
 * Portable CSC matrix algorithms, bound to one engine's dense and indexed Level 1 kernels.
 *
 * Traversal, structure, alias staging and triangle selection are shared across platforms; numerical leaves
 * may group arithmetic differently. The kernels are reached where work is a contiguous run or an indexed
 * slice; [routeOf] is what says which of the two a given call is.
 */
@Suppress("TooManyFunctions") // the sparse BLAS surface
@OptIn(UnsafeKoblasApi::class)
internal class PortableSparseBlas(
    private val vectorKernels: DenseVectorKernels,
    private val indexedKernels: IndexedSparseKernels,
    private val panelKernels: SparsePanelKernels,
) : SparseBlas {
    override fun prepare(a: SparseMatrix): PreparedSparseMatrix = PreparedSparseMatrix(a)

    override fun routeOf(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute =
        SparseRouteReporter(vectorKernels, indexedKernels, panelKernels).routeOf(operation, call)

    @Suppress("LongParameterList") // the BLAS dgemv signature plus the workspace
    override fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        // Zero alpha or zero extent reads neither operand; only destination scaling remains.
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
            applyBeta(vectorKernels, y, 0, y.size, beta)
            return
        }
        // Snapshot aliased operands before destination scaling overwrites their coefficients.
        staged(workspace, x, x === y) { stableX ->
            staged(workspace, a, a.values === y) { stableA ->
                gemvCore(alpha, stableA, stableX, beta, y, transpose)
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dgemv signature, over the snapshots taken for it
    private fun gemvCore(
        alpha: Double,
        stableA: SparseMatrix,
        stableX: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        applyBeta(vectorKernels, y, 0, y.size, beta)
        if (transpose) {
            for (j in 0 until stableA.cols) {
                val start = stableA.colPointers[j]
                val end = stableA.colPointers[j + 1]
                if (start == end) continue
                val sum = indexedKernels.dotDense(
                    stableA.rowIndices,
                    start,
                    stableA.values,
                    start,
                    end - start,
                    stableX,
                )
                y[j] += alpha * sum
            }
        } else {
            for (j in 0 until stableA.cols) {
                val xj = alpha * stableX[j]
                val start = stableA.colPointers[j]
                indexedKernels.axpy(
                    stableA.rowIndices,
                    start,
                    stableA.values,
                    start,
                    stableA.colPointers[j + 1] - start,
                    xj,
                    y,
                )
            }
        }
    }

    override fun transpose(a: SparseMatrix): SparseMatrix = transposeCsc(a)

    @Suppress("LongParameterList") // the BLAS dsymv signature plus the workspace
    override fun symv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSymvOperands(a, x.size, y.size)
        if (alpha == 0.0) {
            applyBeta(vectorKernels, y, 0, y.size, beta)
            return
        }
        staged(workspace, x, x === y) { stableX ->
            staged(workspace, a, a.values === y) { stableA ->
                applyBeta(vectorKernels, y, 0, y.size, beta)
                for (column in 0 until stableA.cols) {
                    panelKernels.symmetricVectorColumn(
                        alpha,
                        column,
                        stableA.rowIndices,
                        stableA.values,
                        stableA.colPointers[column],
                        stableA.colPointers[column + 1],
                        stableX,
                        y,
                        lower,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymm signature plus the workspace
    override fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSymmOperands(a, b, c, right)
        // Return before staging or loans: an empty axis may be paired with an enormous one.
        if (c.rows == 0 || c.cols == 0) return
        if (alpha == 0.0) {
            scaleRegion(c, beta, vectors = vectorKernels)
            return
        }
        staged(workspace, a, a.values === c.values) { stableA ->
            stagedRegion(workspace, b, c) { stableB ->
                scaleRegion(c, beta, vectors = vectorKernels)
                if (right) {
                    for (column in 0 until stableA.cols) {
                        panelKernels.symmetricRightColumn(
                            alpha,
                            column,
                            stableA.rowIndices,
                            stableA.values,
                            stableA.colPointers[column],
                            stableA.colPointers[column + 1],
                            stableB.values,
                            c.values,
                            stableB.rows,
                            lower, stableB.offset, stableB.leadingDimension, c.offset, c.leadingDimension,
                        )
                    }
                } else {
                    multiplySymmetricFromTheLeft(panelKernels, alpha, stableA, stableB, c, lower, workspace)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace
    override fun trsv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        // Snapshot an aliased triangle before substitution overwrites coefficients needed by later columns.
        staged(workspace, a, a.values === x) { stable ->
            trsvCore(panelKernels, stable, x, lower, transpose, unitDiag)
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmv signature plus the workspace
    override fun trmv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        staged(workspace, a, a.values === x) { stable ->
            trmvCore(panelKernels, stable, x, lower, transpose, unitDiag)
        }
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature, plus the side the sparse operand sits on
    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        right: Boolean,
        workspace: Workspace?,
    ) {
        // Right-side multiplication uses the same derivation with operands reversed.
        if (right) {
            requireGemmOperands(b, transposeB, a, transposeA, c)
        } else {
            requireGemmOperands(a, transposeA, b, transposeB, c)
        }
        val m = c.rows
        val n = c.cols
        val k = if (right) {
            if (transposeB) b.rows else b.cols
        } else {
            if (transposeA) a.rows else a.cols
        }
        if (c.rows == 0 || c.cols == 0) return
        if (alpha == 0.0) {
            scaleRegion(c, beta, vectors = vectorKernels)
            return
        }
        staged(workspace, a, a.values === c.values) { stableA ->
            stagedRegion(workspace, b, c) { stableB ->
                scaleRegion(c, beta, vectors = vectorKernels)
                if (right) {
                    multiplyFromTheRight(
                        panelKernels, alpha, stableA, transposeA, stableB, transposeB, c, m, workspace,
                    )
                } else {
                    multiplyFromTheLeft(
                        panelKernels, alpha, stableA, transposeA, stableB, transposeB, c, m, n, k, workspace,
                    )
                }
            }
        }
    }

    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
    ): SparseMatrix {
        requireProductOperands(a, transposeA, b, transposeB, "gemm")
        val aRows = if (transposeA) a.cols else a.rows
        val bCols = if (transposeB) b.rows else b.cols
        // Return before orienting: transpose pointers are row-indexed and may exceed array limits
        // even when the product is empty.
        if (a.nnz == 0 || b.nnz == 0 || aRows == 0 || bCols == 0) return emptyResult(aRows, bCols, "gemm")
        val left = oriented(a, transposeA, alpha != 0.0)
        val right = oriented(b, transposeB, alpha != 0.0)
        return multiplySparse(left, right, alpha)
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature plus the workspace
    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        workspace: Workspace?,
    ) {
        requireGemmOperands(a, transposeA, b, transposeB, c)
        if (alpha == 0.0) {
            scaleRegion(c, beta, vectors = vectorKernels)
            return
        }
        if (a.nnz == 0 || b.nnz == 0) {
            // Only destination scaling remains; avoid row-indexed orientation scratch.
            scaleRegion(c, beta, vectors = vectorKernels)
            return
        }
        staged(workspace, a, a.values === c.values) { stableA ->
            staged(workspace, b, b.values === c.values) { stableB ->
                val left = oriented(stableA, transposeA, true)
                val right = oriented(stableB, transposeB, true)
                scaleRegion(c, beta, vectors = vectorKernels)
                multiplySparseInto(alpha, left, right, c)
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus the workspace
    override fun syrk(
        alpha: Double,
        a: SparseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        requireSyrkOperands(a, transpose, c)
        if (alpha == 0.0) {
            scaleRegion(c, beta, lower)
            return
        }
        if (a.nnz == 0 || n == 0) {
            // An empty pattern needs only triangle scaling, avoiding potentially oversized row-indexed adjacency.
            scaleRegion(c, beta, lower)
            return
        }
        staged(workspace, a, a.values === c.values) { stableA ->
            scaleRegion(c, beta, lower)
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
        requireSameShape(rows, cols, b, "addScaled")
        if (a.nnz == 0 && b.nnz == 0) return emptyResult(rows, cols, "addScaled")
        val left = oriented(a, transposeA, alpha != 0.0)
        // An identical pattern needs neither a structural merge nor room for a second copy of every entry.
        // Different patterns usually disagree at an early row, before every column pointer needs comparing.
        if (left.rowIndices.contentEquals(b.rowIndices) && left.colPointers.contentEquals(b.colPointers)) {
            val values = if (alpha == 0.0) {
                b.values.copyOf()
            } else {
                DoubleArray(left.nnz) { alpha * left.values[it] + b.values[it] }
            }
            return SparseMatrix.wrapTrusted(rows, cols, left.copyColumnPointers(), left.copyRowIndices(), values)
        }
        val union = left.nnz.toLong() + b.nnz
        requireShape(
            union <= Int.MAX_VALUE,
        ) { "scaled addition needs $union stored entries, more than one array can hold" }
        val pointers = IntArray(left.cols + 1)
        val rowIndices = IntArray(union.toInt())
        val values = DoubleArray(rowIndices.size)
        var count = 0
        for (column in 0 until left.cols) {
            count += SparseAccumulationKernels.mergeScaledColumns(
                alpha,
                left.rowIndices,
                left.values,
                left.colPointers[column],
                left.colPointers[column + 1],
                b.rowIndices,
                b.values,
                b.colPointers[column],
                b.colPointers[column + 1],
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
            if (count == rowIndices.size) rowIndices else rowIndices.copyOf(count),
            if (count == values.size) values else values.copyOf(count),
        )
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature plus the workspace
    override fun trsm(
        a: SparseMatrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        requireTriangularMatrixOperands(a, b, right, "trsm")
        val n = a.rows
        if (alpha == 0.0) {
            scaleRegion(b, 0.0)
            return
        }
        if (n == 0) return
        val rightHandSides = if (right) b.rows else b.cols
        if (rightHandSides == 0) return
        // Snapshot an aliased triangle before alpha scales the right-hand sides and changes its coefficients.
        staged(workspace, a, a.values === b.values) { triangle ->
            scaleRegion(b, alpha, vectors = vectorKernels)
            if (right) {
                withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                    trsmRightCore(panelKernels, triangle, b, lower, !transpose, diagonal)
                }
            } else {
                withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                    triangularLeftCore(
                        panelKernels, true, triangle, b, lower, transpose, unitDiag, diagonal, workspace,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature plus the workspace
    override fun trmm(
        a: SparseMatrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        requireTriangularMatrixOperands(a, b, right, "trmm")
        val n = a.rows
        if (alpha == 0.0) {
            scaleRegion(b, 0.0)
            return
        }
        if (n == 0) return
        val rightHandSides = if (right) b.rows else b.cols
        if (rightHandSides == 0) return
        staged(workspace, a, a.values === b.values) { triangle ->
            scaleRegion(b, alpha, vectors = vectorKernels)
            // Unlike trsm, trmm reads the diagonal once per right-hand side.
            withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                if (right) {
                    trmmRightCore(panelKernels, triangle, b, lower, transpose, unitDiag, diagonal)
                } else {
                    triangularLeftCore(
                        panelKernels, false, triangle, b, lower, transpose, unitDiag, diagonal, workspace,
                    )
                }
            }
        }
    }

    /**
     * A CSC result of this shape with nothing stored, for a call whose operands reach no position. [what]
     * names the operation in the one error this raises, a column count no array of pointers can hold.
     */
    private fun emptyResult(rows: Int, cols: Int, what: String): SparseMatrix = SparseMatrix.wrapTrusted(
        rows,
        cols,
        IntArray(pointerLength(cols, what)),
        IntArray(0),
        DoubleArray(0),
    )

    /** `beta · C` over exactly the selected triangle, honoring the zero-beta overwrite convention. */


    /**
     * [a] in the requested orientation, materializing a transpose only where one is asked for.
     *
     * A zero alpha discovers the same structure without reading coefficients, so the transposed pattern is
     * built without copying coefficients: the contract says alpha of zero reads no operand value, and a
     * transpose that copied them would break it before the product ever ran.
     */
    private fun oriented(a: SparseMatrix, transpose: Boolean, readValues: Boolean): SparseMatrix {
        if (!transpose) return a
        return transposeCsc(a, readValues)
    }
}
