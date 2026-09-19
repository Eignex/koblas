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
import com.eignex.koblas.sparse.internal.pointerLength
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

    override fun matrixRouteOf(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute {
        if (operation == SparseMatrixOperation.Prepare) {
            return route(
                operation,
                RouteKind.Direct,
                emptyList(),
                "a snapshot is copied; no arithmetic kernel runs",
            )
        }
        val scaling = destinationScaling(operation, call)
        if (noWork(operation, call)) {
            return route(
                operation,
                RouteKind.NoWork,
                scaling,
                "the call's own contract stops before the arithmetic, so only the destination scaling runs",
            )
        }
        return when (operation) {
            SparseMatrixOperation.Gemv -> scatteredRoute(operation, call, scaling)

            SparseMatrixOperation.GemvTransposed -> route(
                operation,
                RouteKind.Direct,
                scaling + "${ScalarIndexedSparseKernels.name}/${SparseOperation.DotDense.entryPoint}",
                "a transposed CSC reduction is fixed to the ordered scalar dot, which is its accumulation contract",
            )

            SparseMatrixOperation.GemmDenseRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = false,
                "an update whose scaled coefficient is exactly zero is formed by the traversal rather than by " +
                    "the kernel, which skips a zero multiplier",
            )

            SparseMatrixOperation.TrsmRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = false,
                "a stored zero coefficient skips its update and the diagonal is divided by in the traversal",
            )

            SparseMatrixOperation.TrmmRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = true,
                "a stored zero coefficient skips its update",
            )

            else -> route(
                operation,
                RouteKind.Direct,
                scaling,
                "the arithmetic is the CSC traversal's own; beyond any destination scaling no kernel is called",
            )
        }
    }

    private fun route(
        operation: SparseMatrixOperation,
        kind: RouteKind,
        components: List<String>,
        reason: String?,
    ): SparseMatrixRoute =
        SparseMatrixRoute(operation, kind, SPARSE_SCHEDULING, operation.entryPoint, components, reason)

    /**
     * Whether the call's own contract stops before any arithmetic.
     *
     * A zero multiplier, an empty operand or an empty destination all end the call after the destination has
     * been scaled, which is the one component such a call still executes.
     */
    private fun noWork(operation: SparseMatrixOperation, call: SparseCall): Boolean {
        val a = call.matrix
        // An empty destination or an empty inner extent is a call with nothing to compute, whatever its
        // operand holds. A destination the operation does not have is not the same as one with no elements,
        // which is why these arrive as null rather than as zero.
        if (call.destinationElements == 0 || call.depth == 0) return true
        val emptyOperand = a.rows == 0 || a.cols == 0
        return when (operation) {
            // An operation returning a fresh structural result still discovers its pattern when the multiplier
            // is zero. The contract says the operand's values are not read, not that its positions are not
            // found, so this is work even though no coefficient is loaded.
            SparseMatrixOperation.GemmSparse, SparseMatrixOperation.AddScaled,
            SparseMatrixOperation.SyrkSparse, SparseMatrixOperation.Transpose,
            -> emptyOperand

            else -> call.alpha == 0.0 || emptyOperand
        }
    }

    /**
     * The destination scaling component, which is a dense Level 1 `scale` when the multiplier is neither zero
     * nor one.
     *
     * A zero multiplier fills rather than scales and a unit one touches nothing, so neither names a kernel.
     * The selected-triangle rank update writes its own triangle element by element and never reaches one
     * either, which is why it is excluded here rather than left to the multiplier.
     */
    private fun destinationScaling(operation: SparseMatrixOperation, call: SparseCall): List<String> {
        val multiplier = when (operation) {
            SparseMatrixOperation.TrsmLeft, SparseMatrixOperation.TrsmRight,
            SparseMatrixOperation.TrmmLeft, SparseMatrixOperation.TrmmRight,
            -> call.alpha

            SparseMatrixOperation.SyrkDense, SparseMatrixOperation.SyrkSparse,
            SparseMatrixOperation.GemmSparse, SparseMatrixOperation.AddScaled,
            SparseMatrixOperation.Transpose, SparseMatrixOperation.Trsv, SparseMatrixOperation.Trmv,
            -> return emptyList()

            else -> call.beta
        }
        val elements = call.destinationElements ?: return emptyList()
        if (multiplier == 0.0 || multiplier == 1.0 || elements == 0) return emptyList()
        val leaf = vectorKernels.implementationFor(DenseOperation.Scale, elements)
            ?: return listOf("${vectorKernels.name}/scale")
        return listOf("$leaf/scale")
    }

    /**
     * A product that scatters or reduces one CSC column at a time, where the column's own length decides which
     * indexed kernel it reaches.
     *
     * Columns of one matrix need not be the same length, so the implementations they reach need not agree.
     * Naming one of them would publish the other under its label, so a matrix whose columns straddle a
     * crossover is reported as the composition it is.
     */
    private fun scatteredRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
    ): SparseMatrixRoute {
        val leaves = indexedLeaves(call.matrix, SparseOperation.Axpy)
        val entry = SparseOperation.Axpy.entryPoint
        if (leaves.size == 1) {
            return route(
                operation,
                RouteKind.Direct,
                scaling + "${leaves.single()}/$entry",
                "every stored column reaches the same indexed kernel at its own length",
            )
        }
        return route(
            operation,
            RouteKind.Composed,
            scaling + leaves.map { "$it/$entry" },
            "this matrix's column lengths straddle the indexed crossover, so its columns reach " +
                leaves.joinToString(" and "),
        )
    }

    /** The distinct indexed implementations the stored columns of [a] reach for [operation], in first-seen order. */
    private fun indexedLeaves(a: SparseMatrix, operation: SparseOperation): List<String> {
        val leaves = ArrayList<String>(2)
        for (j in 0 until a.cols) {
            val length = a.colPointers[j + 1] - a.colPointers[j]
            val leaf = indexedKernels.implementationFor(operation, length)
            if (leaf !in leaves) leaves.add(leaf)
        }
        return leaves
    }

    /**
     * A call whose unit of work is a whole contiguous dense column, which is where the selected dense kernels
     * run inside a sparse operation.
     *
     * Always composed: the traversal decides per stored entry whether its column update happens at all, so the
     * named kernel serves some units and not others. Saying which kernel it is, and that the traversal keeps
     * the rest, is the whole of the honest answer.
     */
    private fun wholeColumnRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
        multiplies: Boolean,
        skip: String,
    ): SparseMatrixRoute {
        // Composed rather than direct, because the traversal decides per stored entry whether the named
        // kernel is called at all. The component is what this call can reach, not what it is certain to run.
        val axpy = vectorKernels.implementationFor(DenseOperation.Axpy, call.updateRun)
        val components = ArrayList(scaling)
        if (axpy == null) {
            return route(
                operation,
                RouteKind.Composed,
                components,
                "the dense kernels decide axpy on their own values, so no leaf can be named beforehand",
            )
        }
        components.add("$axpy/axpy")
        if (multiplies) {
            val scale = vectorKernels.implementationFor(DenseOperation.Scale, call.updateRun)
            if (scale != null) components.add("$scale/scale")
        }
        return route(operation, RouteKind.Composed, components, skip)
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
        // The no-read shortcut comes first: a zero multiplier or a zero extent reads neither operand, and
        // scaling the destination is all that is left to do.
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
            applyBeta(vectorKernels, y, 0, y.size, beta)
            return
        }
        // Then the snapshots, before the destination is written. Either operand may be the destination's own
        // buffer, and scaling it first would feed the product values the caller never supplied.
        val stableX = if (x === y) x.copyOf() else x
        val stableA = a.stableFor(y)
        applyBeta(vectorKernels, y, 0, y.size, beta)
        if (transpose) {
            for (j in 0 until stableA.cols) {
                val start = stableA.colPointers[j]
                // CSC gemv promises one input-order accumulation chain; a raw dot permits backend reassociation.
                val sum = ScalarIndexedSparseKernels.dotDense(
                    stableA.rowIndices,
                    start,
                    stableA.values,
                    start,
                    stableA.colPointers[j + 1] - start,
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
        // The substitution overwrites x as it goes, so a triangle sharing that buffer is snapshotted first:
        // every column it has yet to reach must still hold the coefficients the caller supplied.
        trsvCore(panelKernels, a.stableFor(x), x, lower, transpose, unitDiag)
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
        // Answered before either operand is oriented, because orienting allocates one pointer per row of the
        // operand and an operand may have more rows than an array can index even when the product it takes
        // part in is empty. Nothing stored on either side reaches no position either.
        if (a.nnz == 0 || b.nnz == 0 || aRows == 0 || bCols == 0) {
            return SparseMatrix.wrapTrusted(
                aRows,
                bCols,
                IntArray(pointerLength(bCols, "gemm")),
                IntArray(0),
                DoubleArray(0),
            )
        }
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
        if (a.nnz == 0 || b.nnz == 0) {
            // No position is reached, so the destination scaling is the whole of the answer and neither
            // operand is oriented. The same reasoning as the fresh sparse product above.
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
        if (a.nnz == 0 || n == 0) {
            // Nothing is stored to reach a position with, so the triangle is scaled and the row adjacency
            // that would have found one is never built. Its scratch is indexed by the source's rows, which a
            // matrix may have more of than an array can hold.
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
        if (a.nnz == 0 && b.nnz == 0) {
            // Neither side contributes a position, so the union is empty and no orientation is built for it.
            return SparseMatrix.wrapTrusted(
                rows,
                cols,
                IntArray(pointerLength(cols, "addScaled")),
                IntArray(0),
                DoubleArray(0),
            )
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
        val rightHandSides = if (right) b.rows else b.cols
        if (rightHandSides == 0) return
        // The triangle is snapshotted before alpha scales the block, not after: a triangle sharing the block's
        // buffer would otherwise be solved against its own scaled coefficients.
        withStableSparse(a, b.values, workspace) { triangle ->
            if (alpha != 1.0) vectorKernels.scale(b.values, 0, alpha, b.values.size)
            if (right) {
                withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                    trsmRightCore(panelKernels, triangle, b, lower, !transpose, diagonal)
                }
            } else {
                workspace.borrow(SPARSE_RHS_WIDTH) { work ->
                    withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                        trsmLeftCore(panelKernels, triangle, b, lower, transpose, diagonal, work)
                    }
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
        if (n == 0) return
        withStableSparse(a, b.values, workspace) { triangle ->
            if (alpha != 1.0) vectorKernels.scale(b.values, 0, alpha, b.values.size)
            // The diagonal is read once for every right-hand side rather than once per column, as trsm does.
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
