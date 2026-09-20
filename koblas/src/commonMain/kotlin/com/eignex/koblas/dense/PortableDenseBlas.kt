@file:Suppress(
    "VariableNaming",
    "FunctionParameterNaming",
    "TooManyFunctions",
    "LongParameterList",
    "MaxLineLength",
)

package com.eignex.koblas.dense

import com.eignex.koblas.*

/**
 * Common Kotlin dense BLAS used by every built-in engine without requiring a host library.
 *
 * This file owns validation, triangle selection, alias staging and which shared schedule a call belongs to.
 * What runs inside a window is a backend's: [panels] owns the arithmetic of a Level 2 window and how many
 * columns of it are worth doing at once, [products] owns the register tile a product block is cut into, and
 * [triangles] owns the substitution over one diagonal block. Neither side knows the other's business: no
 * extent here is a multiple of anything, and no loop below advances by four because a backend once did.
 *
 * Level 3 is scheduled rather than written out. A product goes to shared product scheduling, a selected
 * triangle is the same schedule with the blocks outside it dropped and the ones across the diagonal merged,
 * a symmetric operand is cut into diagonal blocks and the stored strips beside them, and a triangular
 * routine is diagonal substitutions with those windows between them. Which of those a given call reaches,
 * and which body inside it, is what [routeOf] answers rather than letting an engine's name imply it.
 */
internal class PortableDenseBlas(
    private val vectors: DenseVectorKernels,
    private val panels: DensePanelKernels,
    private val products: DenseProductKernels = PortableProductKernels,
    private val triangles: DenseTriangularKernels = PortableTriangularKernels,
) : RoutedDenseBlas {
    private val routes = DenseRouteReporter(vectors, panels, products, triangles)

    private fun scaled(beta: Double, previous: Double): Double = if (beta == 0.0) 0.0 else beta * previous

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        val depth = if (transpose) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            applyBeta(vectors, y, 0, y.size, beta)
            return
        }
        // Snapshots before the destination is touched: either operand may be the destination's own buffer,
        // and scaling it first would feed the product values the caller never supplied. The loan is scoped to
        // the call, which is exactly how long a staged operand is read for.
        staged(workspace, a.values, a.values === y) { stableA ->
            staged(workspace, x, x === y) { stableX ->
                gemvCore(alpha, a, stableA, stableX, beta, y, transpose)
            }
        }
    }

    private fun gemvCore(
        alpha: Double,
        a: DenseMatrix,
        stableA: DoubleArray,
        stableX: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        val rows = a.rows
        if (transpose) {
            // Each output is one column reduced against x, and beta reaches it exactly once, in the panel.
            forEachPanel(a.cols, panels.executionGroup(PanelWork.MultiDot, rows, a.cols)) { start, width ->
                panels.multiDot(alpha, stableA, start * rows, rows, stableX, 0, 1, rows, width, beta, y, start, 1)
            }
        } else {
            applyBeta(vectors, y, 0, y.size, beta)
            forEachPanel(a.cols, panels.executionGroup(PanelWork.ColumnUpdate, rows, a.cols)) { start, width ->
                panels.columnUpdate(alpha, stableA, start * rows, rows, stableX, start, 1, rows, width, y, 0, 1)
            }
        }
    }

    override fun transpose(a: DenseMatrix): DenseMatrix {
        val result = DenseMatrix.zero(a.cols, a.rows)
        for (j in 0 until a.cols) for (i in 0 until a.rows) result.values[j + i * a.cols] = a.values[i + j * a.rows]
        return result
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        requireGemmOperands(a, transposeA, b, transposeB, c)
        // Nothing to write means nothing to stage and nothing to schedule. The extents the scratch would be
        // sized from are the operands', which an empty destination says nothing about, so a product with no
        // output would otherwise borrow a column or a pair of panels for a result that does not exist.
        if (c.values.isEmpty()) return
        val depth = if (transposeA) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scale(c.values, beta)
            return
        }
        staged(workspace, a.values, a.values === c.values) { av ->
            staged(workspace, b.values, b.values === c.values) { bv ->
                gemmCore(alpha, a, transposeA, av, b, transposeB, bv, beta, c, depth, workspace)
            }
        }
    }

    /**
     * The product itself, over operands already staged against an overlap with the destination.
     *
     * Which of the two routes runs is the extents' answer rather than the engine's, and [routeOf] asks the
     * same question of the same numbers.
     */
    private fun gemmCore(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        av: DoubleArray,
        b: DenseMatrix,
        transposeB: Boolean,
        bv: DoubleArray,
        beta: Double,
        c: DenseMatrix,
        depth: Int,
        workspace: Workspace?,
    ) {
        productWindow(
            products, panels, alpha, av, 0, a.rows, transposeA, bv, 0, b.rows, transposeB,
            beta, c.values, 0, c.rows, c.rows, c.cols, depth, OutputTriangle.Full, workspace,
        )
    }

    /**
     * `C = alpha · A · B + beta · C` where both operands are already packed for this backend's tile.
     *
     * The layouts are checked against the tile that will read them and against each other before anything is
     * written, because a panel grouped for another shape or standing in the other operand's position reads
     * neighbouring values as its own and leaves no trace in the result.
     *
     * No staging and no scratch: a packed operand owns storage of its own, so it cannot be the destination.
     */
    fun gemm(alpha: Double, a: PackedMatrix, b: PackedMatrix, beta: Double, c: DenseMatrix) {
        a.layout.requireUsableBy(PackedRole.Left, products.tileRows)
        b.layout.requireUsableBy(PackedRole.Right, products.tileColumns)
        requirePackedProductShape(a.rows, a.columns, b.rows, b.columns, c)
        if (c.values.isEmpty()) return
        if (alpha == 0.0 || a.columns == 0) {
            scale(c.values, beta)
            return
        }
        blockedProduct(
            products, alpha, NO_OPERAND, 0, 0, false, a, NO_OPERAND, 0, 0, false, b,
            beta, c.values, 0, c.rows, c.rows, c.cols, a.columns, OutputTriangle.Full, null,
        )
    }

    /** [gemm] with only the left operand retained; the right is packed for this call from [b]. */
    fun gemm(
        alpha: Double,
        a: PackedMatrix,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        a.layout.requireUsableBy(PackedRole.Left, products.tileRows)
        val rows = if (transposeB) b.cols else b.rows
        val columns = if (transposeB) b.rows else b.cols
        requirePackedProductShape(a.rows, a.columns, rows, columns, c)
        if (c.values.isEmpty()) return
        if (alpha == 0.0 || a.columns == 0) {
            scale(c.values, beta)
            return
        }
        staged(workspace, b.values, b.values === c.values) { bv ->
            blockedProduct(
                products, alpha, NO_OPERAND, 0, 0, false, a, bv, 0, b.rows, transposeB, null,
                beta, c.values, 0, c.rows, c.rows, c.cols, a.columns, OutputTriangle.Full, workspace,
            )
        }
    }

    /** [gemm] with only the right operand retained; the left is packed for this call from [a]. */
    fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: PackedMatrix,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        b.layout.requireUsableBy(PackedRole.Right, products.tileColumns)
        val rows = if (transposeA) a.cols else a.rows
        val columns = if (transposeA) a.rows else a.cols
        requirePackedProductShape(rows, columns, b.rows, b.columns, c)
        if (c.values.isEmpty()) return
        if (alpha == 0.0 || columns == 0) {
            scale(c.values, beta)
            return
        }
        staged(workspace, a.values, a.values === c.values) { av ->
            blockedProduct(
                products, alpha, av, 0, a.rows, transposeA, null, NO_OPERAND, 0, 0, false, b,
                beta, c.values, 0, c.rows, c.rows, c.cols, columns, OutputTriangle.Full, workspace,
            )
        }
    }

    /** `op(A)` packed for the left of a product against this backend's tile. */
    fun packLeft(a: DenseMatrix, transpose: Boolean): PackedMatrix = packedLeft(a, transpose, products.tileRows)

    /** `op(B)` packed for the right of a product against this backend's tile. */
    fun packRight(b: DenseMatrix, transpose: Boolean): PackedMatrix = packedRight(b, transpose, products.tileColumns)

    override fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        // Nothing to write means nothing to stage: the extents a staging loan would be sized from are the
        // operands', which an empty destination says nothing about, and two empty operands that share one
        // empty array would otherwise be staged against each other.
        if (c.values.isEmpty()) return
        val depth = if (transposeA) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleTriangle(c, beta, lower)
            return
        }
        staged(workspace, a.values, a.values === c.values) { av ->
            staged(workspace, b.values, b.values === c.values) { bv ->
                productWindow(
                    products, panels, alpha, av, 0, a.rows, transposeA, bv, 0, b.rows, transposeB,
                    beta, c.values, 0, c.rows, c.rows, c.cols, depth, selectedTriangle(lower), workspace,
                )
            }
        }
    }

    /**
     * `y = alpha · A · x + beta · y` over the stored triangle only, one pass over each column.
     *
     * A stored column of a symmetric matrix is two things at once: the part of a column below the diagonal,
     * and the part of the mirrored row to the left of it. [DensePanelKernels.coupledUpdateDot] does both from
     * one read, and the square corner where a group's columns overlap each other is the remainder that walk
     * cannot cover, which is scalar work here.
     */
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireSymvOperands(a, x.size, y.size)
        val n = a.rows
        if (alpha == 0.0 || n == 0) {
            applyBeta(vectors, y, 0, y.size, beta)
            return
        }
        val av = if (a.values === y) a.values.copyOf() else a.values
        val xv = if (x === y) x.copyOf() else x
        applyBeta(vectors, y, 0, y.size, beta)
        forEachPanel(n, panels.executionGroup(PanelWork.CoupledDotUpdate, n, n)) { start, width ->
            symmetricCorner(alpha, av, n, xv, y, start, width, lower)
            val rows = if (lower) n - (start + width) else start
            if (rows > 0) {
                val window = if (lower) start + width else 0
                panels.coupledUpdateDot(
                    alpha, av, window + start * n, n, xv, window, rows, width, y, window, xv, start, y, start,
                )
            }
        }
    }

    /**
     * The square block where a group's own columns and rows meet, which the shared window cannot reach.
     *
     * Every stored position in it contributes twice, once down its column and once along the mirrored row,
     * except the diagonal which is its own mirror. The arithmetic is the panel's, written out: a coefficient
     * scaled by alpha multiplies down, and the reduction is scaled by alpha on its way into the destination.
     */
    private fun symmetricCorner(
        alpha: Double,
        a: DoubleArray,
        n: Int,
        x: DoubleArray,
        y: DoubleArray,
        start: Int,
        width: Int,
        lower: Boolean,
    ) {
        for (c in start until start + width) {
            val coefficient = alpha * x[c]
            for (i in start until start + width) {
                if (if (lower) i < c else i > c) continue
                val value = a[i + c * n]
                y[i] += coefficient * value
                if (i != c) y[c] += alpha * (value * x[i])
            }
        }
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSymmOperands(a, b, c, right)
        if (c.values.isEmpty()) return
        if (alpha == 0.0) {
            applyBeta(vectors, c.values, 0, c.values.size, beta)
            return
        }
        staged(workspace, a.values, a.values === c.values) { av ->
            staged(workspace, b.values, b.values === c.values) { bv ->
                // Spent once over the whole destination, because every entry of it is accumulated into by
                // several of the windows the symmetric operand is cut into and none of them owns it. It is
                // the Level 1 kernel over the whole buffer rather than a loop of this file's, which is what
                // lets the route name the leaf that runs.
                applyBeta(vectors, c.values, 0, c.values.size, beta)
                symmetricProduct(
                    products, panels, alpha, av, a.rows, lower, bv, b.rows, c.values, c.rows,
                    c.rows, c.cols, right, workspace,
                )
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus the workspace
    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyrkOperands(a, transpose, c)
        productTriangle(alpha, a, a, transpose, beta, c, lower, doubled = false, workspace = workspace)
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyr2kOperands(a, b, transpose, c)
        productTriangle(alpha, a, b, transpose, beta, c, lower, doubled = true, workspace = workspace)
    }

    /**
     * `alpha · op(A) · op(B)ᵀ + beta · C` in one triangle, and the same with the operands swapped when
     * [doubled].
     *
     * A rank-k update is a product whose right operand is the left one transposed, and a rank-2k update is
     * the sum of the two products its definition names. Composing rather than fusing is deliberate: both
     * halves are ordinary windows of shared product scheduling, so each is packed, blocked and tiled like
     * any other product, while a fused traversal would need a second implementation of all of that to beat
     * them together. The local evidence compares the two.
     *
     * `beta` is carried by the first of the two and the second accumulates, which is how it reaches every
     * selected entry exactly once.
     */
    private fun productTriangle(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        doubled: Boolean,
        workspace: Workspace?,
    ) {
        if (c.values.isEmpty()) return
        val depth = if (transpose) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleTriangle(c, beta, lower)
            return
        }
        val selected = selectedTriangle(lower)
        // One operand staged at a time, and a rank update passes the same matrix twice, so the second loan
        // is skipped where both operands are that matrix and the first copy already stands for it.
        staged(workspace, a.values, a.values === c.values) { av ->
            staged(workspace, b.values, b.values === c.values && b !== a) { raw ->
                val bv = if (b === a) av else raw
                productWindow(
                    products, panels, alpha, av, 0, a.rows, transpose, bv, 0, b.rows, !transpose,
                    beta, c.values, 0, c.rows, c.rows, c.cols, depth, selected, workspace,
                )
                if (doubled) {
                    productWindow(
                        products, panels, alpha, bv, 0, b.rows, transpose, av, 0, a.rows, !transpose,
                        1.0, c.values, 0, c.rows, c.rows, c.cols, depth, selected, workspace,
                    )
                }
            }
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        requireGerOperands(x.size, y.size, a)
        if (alpha == 0.0 || a.rows == 0) return
        val xv = if (x === a.values) x.copyOf() else x
        val yv = if (y === a.values) y.copyOf() else y
        forEachPanel(a.cols, panels.executionGroup(PanelWork.RankUpdate, a.rows, a.cols)) { start, width ->
            panels.rankUpdate(alpha, a.values, start * a.rows, a.rows, xv, 0, 1, a.rows, width, yv, start, 1)
        }
    }

    /**
     * `A += alpha · x · xᵀ` over the stored triangle, as one rank update per group plus its corner.
     *
     * Every column of a group stores the rows below the last of them, which is the window they share; what
     * is left is the small triangle between the group's first and last column.
     */
    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyrOperands(a, x.size, "syr")
        val n = a.rows
        if (alpha == 0.0 || n == 0) return
        // A vector sharing the destination's buffer is copied, since the update writes what a later column
        // would otherwise read back as its coefficient.
        val staged = x.values === a.values
        val xv = if (staged) x.toDoubleArray() else x.values
        val origin = if (staged) 0 else x.offset
        val step = if (staged) 1 else x.stride
        forEachPanel(n, panels.executionGroup(PanelWork.RankUpdate, n, n)) { start, width ->
            // The corner is the rows between the group's first and last column, where the columns stop
            // agreeing about which of them are stored; everything past it is the window they share.
            val cornerFirst = if (lower) start else start + 1
            val cornerLast = if (lower) start + width - 1 else start + width
            for (c in start until start + width) {
                val coefficient = alpha * xv[origin + c * step]
                for (i in cornerFirst until cornerLast) {
                    if (if (lower) i < c else i > c) continue
                    a.values[i + c * n] += coefficient * xv[origin + i * step]
                }
            }
            val window = if (lower) start + width - 1 else 0
            val rows = if (lower) n - window else start + 1
            panels.rankUpdate(
                alpha, a.values, window + start * n, n, xv, origin + window * step, step, rows, width,
                xv, origin + start * step, step,
            )
        }
    }

    /**
     * `A += alpha · (x · yᵀ + y · xᵀ)` over the stored triangle.
     *
     * The arithmetic stays here rather than becoming two rank updates through [DensePanelKernels], because
     * the composition was measured and does not win outright: each entry takes both contributions in one
     * read-modify-write where two passes take two, and the composed form was behind at the smaller orders,
     * level in the middle and ahead at the larger ones. Adopting it would mean choosing a crossover from one
     * machine's numbers for a modest best case, so the simpler path is kept; the comparison is in the stage
     * evidence.
     */
    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyr2Operands(a, x.size, y.size, "syr2")
        val n = a.rows
        if (alpha == 0.0 || n == 0) return
        val stagedX = x.values === a.values
        val stagedY = y.values === a.values
        val xv = if (stagedX) x.toDoubleArray() else x.values
        val yv = if (stagedY) y.toDoubleArray() else y.values
        val xOrigin = if (stagedX) 0 else x.offset
        val xStep = if (stagedX) 1 else x.stride
        val yOrigin = if (stagedY) 0 else y.offset
        val yStep = if (stagedY) 1 else y.stride
        for (c in 0 until n) {
            val first = if (lower) c else 0
            val last = if (lower) n else c + 1
            val xc = xv[xOrigin + c * xStep]
            val yc = yv[yOrigin + c * yStep]
            for (i in first until last) {
                a.values[i + c * n] += alpha * (xv[xOrigin + i * xStep] * yc + yv[yOrigin + i * yStep] * xc)
            }
        }
    }

    /**
     * Solve `op(T) · x = b` in place, one stored column at a time.
     *
     * Untransposed, a finished entry is divided by its diagonal and then removed from every entry that
     * depends on it, which is one column update. Transposed, an entry is finished by reducing the column
     * against the entries already done, which is one multi-dot. Either way the dependency order is the
     * substitution's and cannot be grouped away; what a panel covers is the independent arithmetic within
     * one step.
     */
    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        val n = a.rows
        val av = if (a.values === x) a.values.copyOf() else a.values
        val forward = lower != transpose
        for (step in 0 until n) {
            val j = if (forward) step else n - 1 - step
            if (transpose) {
                val rows = if (forward) j else n - 1 - j
                val window = if (forward) 0 else j + 1
                if (rows > 0) panels.multiDot(-1.0, av, window + j * n, n, x, window, 1, rows, 1, 1.0, x, j, 1)
                if (!unitDiag) x[j] = x[j] / av[j + j * n]
            } else {
                if (!unitDiag) x[j] = x[j] / av[j + j * n]
                val rows = if (forward) n - 1 - j else j
                val window = if (forward) j + 1 else 0
                if (rows > 0) panels.columnUpdate(-1.0, av, window + j * n, n, x, j, 1, rows, 1, x, window, 1)
            }
        }
    }

    /**
     * `x = op(T) · x` in place, one stored column at a time and in the order that leaves no entry needed
     * after it has been overwritten.
     *
     * Untransposed, a column's entries multiply the entry on its diagonal into every row below it, so the
     * columns are walked away from the diagonal; transposed, an entry is the reduction of its own column
     * against entries not yet touched. Neither direction needs a copy of the input.
     */
    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        val n = a.rows
        val av = if (a.values === x) a.values.copyOf() else a.values
        // Untransposed, a column is consumed before the columns it would overwrite; transposed, a column is
        // produced from entries later columns have not reached yet. The two run in opposite directions.
        val ascending = transpose == lower
        for (step in 0 until n) {
            val j = if (ascending) step else n - 1 - step
            val rows = if (lower) n - 1 - j else j
            val window = if (lower) j + 1 else 0
            if (transpose) {
                if (!unitDiag) x[j] = av[j + j * n] * x[j]
                if (rows > 0) panels.multiDot(1.0, av, window + j * n, n, x, window, 1, rows, 1, 1.0, x, j, 1)
            } else {
                if (rows > 0) panels.columnUpdate(1.0, av, window + j * n, n, x, j, 1, rows, 1, x, window, 1)
                if (!unitDiag) x[j] = av[j + j * n] * x[j]
            }
        }
    }

    override fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true, workspace = workspace)

    override fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false, workspace = workspace)

    /**
     * Both triangular matrix routines, over a triangle staged against an overlap with the block it works on.
     *
     * The staging comes first and the scaling second. `alpha` is spent on the right-hand sides before the
     * substitution, and where the triangle shares their buffer that write would otherwise reach the
     * coefficients the substitution is about to read.
     */
    @Suppress("LongParameterList") // the BLAS dtrsm signature, which of the two it is, and the workspace
    private fun triangularMatrix(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        solve: Boolean,
        workspace: Workspace?,
    ) {
        val what = if (solve) "trsm" else "trmm"
        requireTriangularMatrixOperands(a, b, right, what)
        if (alpha == 0.0) {
            b.values.fill(0.0)
            return
        }
        val sides = if (right) b.rows else b.cols
        val order = a.rows
        if (sides == 0 || order == 0) return
        staged(workspace, a.values, a.values === b.values) { av ->
            triangularMatrix(
                triangles, products, panels, vectors, av, order, b.values, b.rows, sides,
                lower, transpose, unitDiag, right, alpha, solve, workspace,
            )
        }
    }

    private fun scale(values: DoubleArray, beta: Double) {
        if (beta == 0.0) values.fill(0.0) else for (i in values.indices) values[i] *= beta
    }

    private fun scaleTriangle(c: DenseMatrix, beta: Double, lower: Boolean) = forTriangle(c.rows, lower) { i, j ->
        val index = i + j * c.rows
        c.values[index] = scaled(beta, c.values[index])
    }

    private inline fun forTriangle(n: Int, lower: Boolean, action: (Int, Int) -> Unit) {
        for (j in 0 until n) for (i in 0 until n) if ((lower && i >= j) || (!lower && i <= j)) action(i, j)
    }

    /**
     * What a call of [operation] with the facts in [call] executes.
     *
     * Asked of the same backends this engine computes with, so a route is a description of what would run
     * rather than of what an engine of this shape might run.
     */
    override fun routeOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute =
        routes.routeOf(operation, call)
}

/** The part of a square destination a selected-triangle routine writes. */
internal fun selectedTriangle(lower: Boolean): OutputTriangle =
    if (lower) OutputTriangle.Lower else OutputTriangle.Upper

/** The component name every built-in dense Level 2 and 3 call reports, whatever panels it calls. */
internal const val DENSE_SCHEDULING: String = "portable-dense"

/** The component name the copy into packed tile groups reports. The packers are portable on every engine. */
internal const val PRODUCT_PACKING: String = "portable-pack"

/**
 * The component name the selected-triangle writeback reports.
 *
 * Its own name because it is arithmetic the tile did not do: a block straddling the diagonal accumulates
 * whole tiles and then keeps part of each, and a route that named only the tile would claim the whole of it
 * reached the destination.
 */
internal const val TRIANGLE_SELECTION: String = "portable-select"

/** The component name the copy of one diagonal block of a symmetric operand into a full square reports. */
internal const val SYMMETRIC_EXPANSION: String = "portable-mirror"

/** The component name the copy of a block of right-hand sides into adjacent storage reports. */
internal const val TRIANGULAR_GATHER: String = "portable-gather"

/** The entry points whose route is a product block schedule rather than a Level 2 panel one. */
internal val PRODUCT_OPERATIONS: Set<DenseMatrixOperation> = setOf(
    DenseMatrixOperation.Gemm,
    DenseMatrixOperation.GemmPacked,
    DenseMatrixOperation.GemmPackedLeft,
    DenseMatrixOperation.GemmPackedRight,
)

/** The entry points whose route is a product into one triangle of a square destination. */
internal val TRIANGLE_PRODUCT_OPERATIONS: Set<DenseMatrixOperation> = setOf(
    DenseMatrixOperation.Gemmt,
    DenseMatrixOperation.Syrk,
    DenseMatrixOperation.Syr2k,
)

/** The entry points whose route is a triangular block schedule rather than a product or a panel one. */
internal val TRIANGULAR_MATRIX_OPERATIONS: Set<DenseMatrixOperation> = setOf(
    DenseMatrixOperation.Trmm,
    DenseMatrixOperation.Trsm,
)

/** Stands in for an operand that arrived packed, so no unpacked storage is read for that side. */
internal val NO_OPERAND: DoubleArray = DoubleArray(0)

/** The shapes a product between packed and dense operands needs, checked before the destination is touched. */
internal fun requirePackedProductShape(rows: Int, depth: Int, rightRows: Int, columns: Int, c: DenseMatrix) {
    requireShape(depth == rightRows) { "gemm: inner dimensions differ, $depth vs $rightRows" }
    requireShape(c.rows == rows && c.cols == columns) {
        "gemm: destination must be ${rows}x$columns, got ${c.rows}x${c.cols}"
    }
}
