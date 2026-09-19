@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.MatrixWorkspace
import com.eignex.koblas.PreparedSparseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.borrow
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.requireGemmShape
import com.eignex.koblas.requireGemvShape
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSquare
import com.eignex.koblas.requireTriangularMatrixShape
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
import com.eignex.koblas.vendor.RouteKind

/** The component name every built-in sparse Level 2 and 3 call reports, whatever Level 1 kernels it calls. */
internal const val SPARSE_SCHEDULING: String = "portable-csc"

/**
 * Portable CSC matrix algorithms, bound to one engine's dense and indexed Level 1 kernels.
 *
 * Traversal, structure, alias staging, triangle selection and arithmetic order are written here and are the
 * same on every platform. The kernels are reached only where a unit of work is a contiguous run or an indexed
 * slice; [matrixRouteOf] is what says which of the two a given call is.
 */
@Suppress("TooManyFunctions") // the sparse BLAS surface
@OptIn(UnsafeKoblasApi::class)
internal class SparseAlgorithms(
    private val vectorKernels: DenseVectorKernels,
    private val indexedKernels: IndexedSparseKernels,
    private val panelKernels: SparsePanelKernels,
) : SparseBlas {
    override fun prepare(a: SparseMatrix): PreparedSparseMatrix = PreparedSparseMatrix(a, this)

    override fun matrixRouteOf(
        operation: SparseMatrixOperation,
        entriesPerColumn: Int,
        denseRun: Int,
    ): SparseMatrixRoute {
        require(entriesPerColumn >= 0) { "negative stored entry count" }
        require(denseRun >= 0) { "negative dense run length" }
        return when (operation) {
            SparseMatrixOperation.Gemv -> indexedRoute(
                operation,
                SparseOperation.Axpy,
                entriesPerColumn,
                "each CSC column is scattered into the destination by the selected indexed axpy",
            )

            SparseMatrixOperation.GemvTransposed -> SparseMatrixRoute(
                operation,
                RouteKind.Direct,
                SPARSE_SCHEDULING,
                operation.entryPoint,
                "${ScalarIndexedSparseKernels.name}/${SparseOperation.DotDense.entryPoint}",
                "a transposed CSC reduction is fixed to the ordered scalar dot, which is its accumulation contract",
            )

            SparseMatrixOperation.GemmDenseRight,
            SparseMatrixOperation.TrsmRight,
            SparseMatrixOperation.TrmmRight,
            -> denseRoute(operation, denseRun)

            else -> SparseMatrixRoute(
                operation,
                RouteKind.Direct,
                SPARSE_SCHEDULING,
                operation.entryPoint,
                null,
                "the arithmetic is the CSC traversal's own; no Level 1 kernel is called",
            )
        }
    }

    /** A route whose unit of work is one indexed slice, so the selected sparse kernels decide it. */
    private fun indexedRoute(
        operation: SparseMatrixOperation,
        leafOperation: SparseOperation,
        entriesPerColumn: Int,
        reason: String,
    ): SparseMatrixRoute {
        val leaf = indexedKernels.implementationFor(leafOperation, entriesPerColumn)
        return SparseMatrixRoute(
            operation,
            RouteKind.Direct,
            SPARSE_SCHEDULING,
            operation.entryPoint,
            "$leaf/${leafOperation.entryPoint}",
            reason,
        )
    }

    /** A route whose unit of work is one contiguous dense column, so the dense Level 1 kernels decide it. */
    private fun denseRoute(operation: SparseMatrixOperation, denseRun: Int): SparseMatrixRoute {
        val leaf = panelKernels.denseLeaf(DenseOperation.Axpy, denseRun)
            ?: return SparseMatrixRoute(
                operation,
                RouteKind.Composed,
                SPARSE_SCHEDULING,
                operation.entryPoint,
                null,
                "the dense kernels decide axpy on their own values, so no leaf is named beforehand",
            )
        return SparseMatrixRoute(
            operation,
            RouteKind.Direct,
            SPARSE_SCHEDULING,
            operation.entryPoint,
            "$leaf/axpy",
            "the sparse operand on the right makes every update one whole dense column",
        )
    }

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
                val start = a.colPointers[j]
                // CSC gemv promises one input-order accumulation chain; a raw dot permits backend reassociation.
                val sum = ScalarIndexedSparseKernels.dotDense(
                    a.rowIndices,
                    start,
                    a.values,
                    start,
                    a.colPointers[j + 1] - start,
                    x,
                )
                y[j] += alpha * sum
            }
        } else {
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                val start = a.colPointers[j]
                indexedKernels.axpy(a.rowIndices, start, a.values, start, a.colPointers[j + 1] - start, xj, y)
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

    @Suppress("LongParameterList") // the BLAS dsymm signature plus the workspace
    override fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        workspace: MatrixWorkspace?,
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
            applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
            return
        }
        withStableSparse(a, c.values, workspace) { stableA ->
            withStableDense(b, c.values, workspace) { stableB ->
                applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
                for (column in 0 until stableA.cols) {
                    if (right) {
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
                            lower,
                        )
                    } else {
                        panelKernels.symmetricLeftColumn(
                            alpha,
                            column,
                            stableA.rowIndices,
                            stableA.values,
                            stableA.colPointers[column],
                            stableA.colPointers[column + 1],
                            stableB.values,
                            c.values,
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
        requireShape(x.size == a.rows) { "trsv: x length ${x.size} != ${a.rows}" }
        trsvCore(panelKernels, a, x, lower, transpose, unitDiag)
    }

    override fun trmv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trmv")
        requireShape(x.size == a.rows) { "trmv: x length ${x.size} != ${a.rows}" }
        trmvCore(panelKernels, a.stableFor(x), x, lower, transpose, unitDiag)
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
        workspace: MatrixWorkspace?,
    ) {
        // Multiplying the dense operand by the sparse one from the right is this product with the operands
        // the other way round, so the same derivation answers both.
        val (m, k, n) = if (right) {
            requireGemmShape(b, transposeB, a, transposeA, c)
        } else {
            requireGemmShape(a, transposeA, b, transposeB, c)
        }
        if (alpha == 0.0) {
            applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
            return
        }
        withStableSparse(a, c.values, workspace) { stableA ->
            withStableDense(b, c.values, workspace) { stableB ->
                applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
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
        val aRows = if (transposeA) a.cols else a.rows
        val aCols = if (transposeA) a.rows else a.cols
        val bRows = if (transposeB) b.cols else b.rows
        val bCols = if (transposeB) b.rows else b.cols
        requireShape(aCols == bRows) { "gemm: op(A) is ${aRows}x$aCols but op(B) is ${bRows}x$bCols" }
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
        c: DenseMatrix,
        workspace: MatrixWorkspace?,
    ) {
        requireGemmShape(a, transposeA, b, transposeB, c)
        if (alpha == 0.0) {
            applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
            return
        }
        withStableSparse(a, c.values, workspace) { stableA ->
            withStableSparse(b, c.values, workspace) { stableB ->
                val left = oriented(stableA, transposeA, true)
                val right = oriented(stableB, transposeB, true)
                applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
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
        c: DenseMatrix,
        lower: Boolean,
        workspace: MatrixWorkspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        if (alpha == 0.0) {
            scaleTriangle(c, n, beta, lower)
            return
        }
        withStableSparse(a, c.values, workspace) { stableA ->
            scaleTriangle(c, n, beta, lower)
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
            rowIndices.copyOf(count),
            values.copyOf(count),
        )
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature plus the workspace
    override fun trsm(
        a: SparseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: MatrixWorkspace?,
    ) {
        val n = requireTriangularMatrixShape(a, b, right, "trsm")
        if (alpha == 0.0) {
            b.values.fill(0.0)
            return
        }
        if (n == 0) return
        if (alpha != 1.0) vectorKernels.scale(b.values, 0, alpha, b.values.size)
        val rightHandSides = if (right) b.rows else b.cols
        if (rightHandSides == 0) return
        if (right) {
            withExplicitDiagonal(a, n, unitDiag, workspace) { diagonal ->
                trsmRightCore(panelKernels, a, b, lower, !transpose, diagonal)
            }
        } else {
            workspace.borrow(SPARSE_RHS_WIDTH) { work ->
                withExplicitDiagonal(a, n, unitDiag, workspace) { diagonal ->
                    trsmLeftCore(panelKernels, a, b, lower, transpose, diagonal, work)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature plus the workspace
    override fun trmm(
        a: SparseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: MatrixWorkspace?,
    ) {
        val n = requireTriangularMatrixShape(a, b, right, "trmm")
        if (alpha == 0.0) {
            b.values.fill(0.0)
            return
        }
        val triangle = a.stableFor(b.values)
        if (alpha != 1.0) vectorKernels.scale(b.values, 0, alpha, b.values.size)
        if (n == 0) return
        // Read once for every right-hand side rather than once per trmv call, as trsm does.
        withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
            if (right) {
                trmmRightCore(panelKernels, triangle, b, lower, transpose, unitDiag, diagonal)
            } else {
                workspace.borrow(SPARSE_RHS_WIDTH) { work ->
                    trmmLeftCore(panelKernels, triangle, b, lower, transpose, unitDiag, diagonal, work)
                }
            }
        }
    }

    /** `beta · C` over exactly the selected triangle, honoring the zero-beta overwrite convention. */
    private fun scaleTriangle(c: DenseMatrix, n: Int, beta: Double, lower: Boolean) {
        if (beta == 1.0) return
        for (j in 0 until n) {
            val first = if (lower) j else 0
            val last = if (lower) n else j + 1
            for (i in first until last) {
                val at = i + j * n
                c.values[at] = if (beta == 0.0) 0.0 else beta * c.values[at]
            }
        }
    }

    /**
     * [a] in the requested orientation, materializing a transpose only where one is asked for.
     *
     * A zero alpha discovers the same structure without reading coefficients, so the transposed pattern is
     * built over zeroed values rather than over the caller's: the contract says alpha of zero reads no
     * operand value, and a transpose that copied them would break it before the product ever ran.
     */
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
