@file:Suppress(
    "VariableNaming",
    "FunctionParameterNaming",
    "TooManyFunctions",
    "LongParameterList",
    "MaxLineLength",
)

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.vendor.*

/**
 * Common Kotlin dense BLAS used by every built-in engine without requiring a host library.
 *
 * This file owns validation, windows, triangle selection, dependency order and alias staging; [panels] owns
 * the arithmetic inside a window and how many columns of it are worth doing at once. Neither knows the
 * other's business: no extent here is a multiple of anything, and no loop below advances by four because a
 * backend once did.
 *
 * Level 3 is direct scalar traversal here and calls no panel, which [routeOf] reports rather than implying
 * otherwise from the engine's name.
 */
internal class PortableDenseBlas(private val vectors: DenseVectorKernels, private val panels: DensePanelKernels) :
    DenseBlas {
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

    /**
     * [block] with a copy of [values] when [aliased], and with [values] itself when it is not.
     *
     * The copy is a loan for the duration of the call, so a caller repeating one shape reuses the same
     * staging buffer instead of allocating one each time. Where nothing is aliased there is no loan at all,
     * which is what keeps an ordinary call from touching the workspace.
     */
    private inline fun <T> staged(
        workspace: Workspace?,
        values: DoubleArray,
        aliased: Boolean,
        block: (DoubleArray) -> T,
    ): T {
        if (!aliased) return block(values)
        return workspace.borrow(values.size) { copy ->
            values.copyInto(copy)
            block(copy)
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
        val depth = if (transposeA) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scale(c.values, beta)
            return
        }
        staged(workspace, a.values, a.values === c.values) { av ->
            staged(workspace, b.values, b.values === c.values) { bv ->
                gemmCore(alpha, a, transposeA, av, b, transposeB, bv, beta, c, depth)
            }
        }
    }

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
    ) {
        val m = c.rows
        for (j in 0 until c.cols) {
            for (i in 0 until m) {
                var sum = 0.0
                for (p in 0 until depth) {
                    val left = if (transposeA) av[p + i * a.rows] else av[i + p * a.rows]
                    val right = if (transposeB) bv[j + p * b.rows] else bv[p + j * b.rows]
                    sum += left * right
                }
                val index = i + j * m
                c.values[index] = alpha * sum + scaled(beta, c.values[index])
            }
        }
    }

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
        requireGemmtOperands(a, transposeA, b, transposeB, c, symmetricStructure(lower))
        val depth = if (transposeA) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleTriangle(c, beta, lower)
            return
        }
        staged(workspace, a.values, a.values === c.values) { av ->
            staged(workspace, b.values, b.values === c.values) { bv ->
                gemmtCore(alpha, a, transposeA, av, b, transposeB, bv, beta, c, lower, depth)
            }
        }
    }

    @Suppress("LongParameterList") // the gemmt signature, both staged operands and the resolved depth
    private fun gemmtCore(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        av: DoubleArray,
        b: DenseMatrix,
        transposeB: Boolean,
        bv: DoubleArray,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        depth: Int,
    ) {
        forTriangle(c.rows, lower) { i, j ->
            var sum = 0.0
            for (p in 0 until depth) {
                val left = if (transposeA) av[p + i * a.rows] else av[i + p * a.rows]
                val right = if (transposeB) bv[j + p * b.rows] else bv[p + j * b.rows]
                sum += left * right
            }
            val index = i + j * c.rows
            c.values[index] = alpha * sum + scaled(beta, c.values[index])
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
        requireSymvOperands(a, symmetricStructure(lower), x.size, y.size)
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
        requireSymmOperands(a, symmetricStructure(lower), b, c, right)
        if (alpha == 0.0) {
            scale(c.values, beta)
            return
        }
        staged(workspace, a.values, a.values === c.values) { av ->
            staged(workspace, b.values, b.values === c.values) { bv ->
                symmCore(alpha, a, av, b, bv, beta, c, lower, right)
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymm signature plus both staged operands
    private fun symmCore(
        alpha: Double,
        a: DenseMatrix,
        av: DoubleArray,
        b: DenseMatrix,
        bv: DoubleArray,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        for (j in 0 until c.cols) {
            for (i in 0 until c.rows) {
                var sum = 0.0
                val depth = if (right) c.cols else c.rows
                for (p in 0 until depth) {
                    sum += if (right) {
                        bv[i + p * b.rows] * symmetric(av, a.rows, p, j, lower)
                    } else {
                        symmetric(av, a.rows, i, p, lower) * bv[p + j * b.rows]
                    }
                }
                val index = i + j * c.rows
                c.values[index] = alpha * sum + scaled(beta, c.values[index])
            }
        }
    }

    private fun symmetric(values: DoubleArray, n: Int, i: Int, j: Int, lower: Boolean): Double =
        if (lower == (i >= j)) values[i + j * n] else values[j + i * n]

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
        requireSyrkOperands(a, transpose, c, symmetricStructure(lower))
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
        requireSyr2kOperands(a, b, transpose, c, symmetricStructure(lower))
        productTriangle(alpha, a, b, transpose, beta, c, lower, doubled = true, workspace = workspace)
    }

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
        val depth = if (transpose) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleTriangle(c, beta, lower)
            return
        }
        // One operand staged at a time, and a rank update passes the same matrix twice, so the second loan
        // is skipped where both operands are that matrix and the first copy already stands for it.
        staged(workspace, a.values, a.values === c.values) { av ->
            staged(workspace, b.values, b.values === c.values && b !== a) { bv ->
                productTriangleCore(alpha, a, av, b, if (b === a) av else bv, transpose, beta, c, lower, doubled, depth)
            }
        }
    }

    @Suppress("LongParameterList") // the rank-update signature, both staged operands and the resolved depth
    private fun productTriangleCore(
        alpha: Double,
        a: DenseMatrix,
        av: DoubleArray,
        b: DenseMatrix,
        bv: DoubleArray,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        doubled: Boolean,
        depth: Int,
    ) {
        forTriangle(c.rows, lower) { i, j ->
            var sum = 0.0
            for (p in 0 until depth) {
                val aip = if (transpose) av[p + i * a.rows] else av[i + p * a.rows]
                val ajp = if (transpose) av[p + j * a.rows] else av[j + p * a.rows]
                if (doubled) {
                    val bip = if (transpose) bv[p + i * b.rows] else bv[i + p * b.rows]
                    val bjp = if (transpose) bv[p + j * b.rows] else bv[j + p * b.rows]
                    sum += aip * bjp + bip * ajp
                } else {
                    sum += aip * ajp
                }
            }
            val index = i + j * c.rows
            c.values[index] = alpha * sum + scaled(beta, c.values[index])
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
        requireSyrOperands(a, symmetricStructure(lower), "syr", x)
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
     * the composition was measured and does not win outright. Each entry takes both contributions in one
     * read-modify-write where two passes take two, and the composed form was behind at the smaller orders,
     * level in the middle and ahead at the larger ones. Adopting it would mean choosing a crossover from one
     * machine's numbers for a modest best case, so the simpler path is kept; the comparison is in the stage
     * evidence.
     */
    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyr2Operands(a, symmetricStructure(lower), "syr2", x, y)
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

    private fun triangular(values: DoubleArray, n: Int, i: Int, j: Int, transpose: Boolean, unitDiag: Boolean): Double {
        if (unitDiag && i == j) return 1.0
        return if (transpose) values[j + i * n] else values[i + j * n]
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
        requireTriangularVectorOperands(a, triangle(lower, unitDiag), x.size, "trsv")
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
        requireTriangularVectorOperands(a, triangle(lower, unitDiag), x.size, "trmv")
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
     * One right-hand side at a time, over a staged triangle and a borrowed copy of that side.
     *
     * The right-hand side is gathered into its own array because the dependency chain runs down it while the
     * block is stored the other way round for a right-side call. Both that gather and the triangle's staging
     * are loans, so a caller repeating one shape pays for them once.
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
        requireTriangularMatrixOperands(a, triangle(lower, unitDiag), b, right, what)
        if (alpha == 0.0) {
            b.values.fill(0.0)
            return
        }
        val sides = if (right) b.rows else b.cols
        val order = a.rows
        if (sides == 0 || order == 0) return
        staged(workspace, a.values, a.values === b.values) { av ->
            workspace.borrow(order) { rhs ->
                for (s in 0 until sides) {
                    for (k in 0 until order) {
                        rhs[k] = alpha * if (right) b.values[s + k * b.rows] else b.values[k + s * b.rows]
                    }
                    val flipped = if (right) !transpose else transpose
                    if (solve) {
                        solve(av, order, rhs, lower, flipped, unitDiag)
                    } else {
                        multiplyTriangle(av, order, rhs, lower, flipped, unitDiag, workspace)
                    }
                    for (k in 0 until order) {
                        if (right) b.values[s + k * b.rows] = rhs[k] else b.values[k + s * b.rows] = rhs[k]
                    }
                }
            }
        }
    }

    private fun solve(a: DoubleArray, n: Int, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        val forward = lower != transpose
        val order = if (forward) 0 until n else n - 1 downTo 0
        for (i in order) {
            var sum = x[i]
            val inner = if (forward) 0 until i else i + 1 until n
            for (j in inner) sum -= triangular(a, n, i, j, transpose, unitDiag) * x[j]
            x[i] = sum / triangular(a, n, i, i, transpose, unitDiag)
        }
    }

    /** `x = op(T) · x` over one right-hand side, reading a borrowed copy of it as the source. */
    @Suppress("LongParameterList") // the triangle, its three flags and the workspace
    private fun multiplyTriangle(
        a: DoubleArray,
        n: Int,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        workspace.borrow(n) { source ->
            x.copyInto(source, 0, 0, n)
            for (i in 0 until n) {
                var sum = 0.0
                for (j in 0 until n) {
                    val inTriangle = if (lower != transpose) j <= i else j >= i
                    if (inTriangle) sum += triangular(a, n, i, j, transpose, unitDiag) * source[j]
                }
                x[i] = sum
            }
        }
    }

    private fun triangle(lower: Boolean, unitDiag: Boolean): MatrixStructure = when {
        unitDiag && lower -> MatrixStructure.UnitLower
        unitDiag -> MatrixStructure.UnitUpper
        lower -> MatrixStructure.TriangularLower
        else -> MatrixStructure.TriangularUpper
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
     * Derived from the decisions the call itself makes rather than from its extents: the same grouping the
     * traversal will ask for, and the panel implementation each window of that traversal reaches at its own
     * length. The distinction matters where the windows are not all alike. A triangular traversal over four
     * columns cuts windows of three, two, one and nothing, so on a machine whose lane block is four it never
     * reaches a vector body at all, and a route derived from the order alone would say it did.
     */
    fun routeOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        if (call.alpha == 0.0 || call.rows == 0 || call.columns == 0 || call.depth == 0) {
            return route(
                operation,
                RouteKind.NoWork,
                destinationScaling(operation, call, working = false),
                0,
                "the call's own contract stops before the arithmetic, so only the destination scaling runs",
            )
        }
        val scaling = destinationScaling(operation, call, working = true)
        val work = panelWorkOf(operation)
            ?: return route(
                operation,
                RouteKind.Direct,
                scaling,
                0,
                "the arithmetic is this traversal's own; no panel or Level 1 kernel is called",
            )
        return panelRoute(operation, call, scaling, work)
    }

    /** The panel a call of [operation] schedules, or null where it schedules none. */
    private fun panelWorkOf(operation: DenseMatrixOperation): PanelWork? = when (operation) {
        DenseMatrixOperation.Gemv, DenseMatrixOperation.Trmv, DenseMatrixOperation.Trsv -> PanelWork.ColumnUpdate

        DenseMatrixOperation.GemvTransposed, DenseMatrixOperation.TrmvTransposed,
        DenseMatrixOperation.TrsvTransposed,
        -> PanelWork.MultiDot

        DenseMatrixOperation.Symv -> PanelWork.CoupledDotUpdate

        DenseMatrixOperation.Ger, DenseMatrixOperation.Syr -> PanelWork.RankUpdate

        else -> null
    }

    private fun route(
        operation: DenseMatrixOperation,
        kind: RouteKind,
        components: List<String>,
        group: Int,
        reason: String?,
    ): DenseMatrixRoute =
        DenseMatrixRoute(operation, kind, DENSE_SCHEDULING, operation.entryPoint, components, group, reason)

    /**
     * The destination scaling component, which is a dense Level 1 `scale` when the multiplier is neither
     * zero nor one.
     *
     * Which calls reach a kernel for it depends on whether the call does any work. An ordinary transposed
     * matrix-vector product folds beta into the panel that writes each output, so it names no scaling; the
     * same call with nothing to compute has no panel to fold it into and scales the destination through the
     * kernel like the rest. The Level 3 routines scale their selected region with their own loop either way.
     */
    private fun destinationScaling(operation: DenseMatrixOperation, call: DenseCall, working: Boolean): List<String> {
        val scales = when (operation) {
            DenseMatrixOperation.Gemv, DenseMatrixOperation.Symv -> true
            DenseMatrixOperation.GemvTransposed -> !working
            else -> false
        }
        val elements = if (operation == DenseMatrixOperation.GemvTransposed) call.columns else call.rows
        if (!scales || call.beta == 0.0 || call.beta == 1.0 || elements == 0) return emptyList()
        val leaf = vectors.implementationFor(DenseOperation.Scale, elements)
            ?: return listOf("${vectors.name}/scale")
        return listOf("$leaf/scale")
    }

    /**
     * A call whose arithmetic is panels of [work], reported over the windows its traversal will actually cut.
     *
     * A rectangular operand's windows are all as long as it is tall, so one implementation serves the whole
     * call. A triangular or symmetric traversal's shrink towards the diagonal, so which of a backend's bodies
     * they reach can differ between them and can be none of them, and the route says which of the three it is.
     */
    private fun panelRoute(
        operation: DenseMatrixOperation,
        call: DenseCall,
        scaling: List<String>,
        work: PanelWork,
    ): DenseMatrixRoute {
        val group = panels.executionGroup(work, call.rows, call.columns)
        val entry = panelEntryPoint(work)
        val leaves = ArrayList<String>(2)
        forEachWindow(operation, call, group) { rows, width ->
            if (rows > 0) {
                val leaf = panels.implementationFor(work, rows, width, call.contiguous)
                if (leaf !in leaves) leaves.add(leaf)
            }
        }
        return when (leaves.size) {
            // No panel runs, so there is no grouping to report either: the contract says zero where a
            // call schedules none, and a recommendation nothing asked for is not one the call used.
            0 -> route(
                operation,
                RouteKind.Direct,
                scaling,
                0,
                "every window this traversal cuts is empty, so no panel runs and the arithmetic is its own",
            )

            1 -> route(
                operation,
                RouteKind.Direct,
                scaling + "${leaves.single()}/$entry",
                group,
                "every window of this traversal reaches the same panel implementation at its own length",
            )

            else -> route(
                operation,
                RouteKind.Composed,
                scaling + leaves.map { "$it/$entry" },
                group,
                "this traversal's windows shrink towards the diagonal, so they reach " +
                    leaves.joinToString(" and ") + ", and its corner is the traversal's own arithmetic",
            )
        }
    }

    /**
     * The windows a call of [operation] hands to its panel, as the length of each and how many columns it
     * carries.
     *
     * Written out beside the traversals above rather than derived from them, so that a route and a call can
     * be compared with each other. The lengths are what a triangle's storage leaves at each group, which is
     * why the selected triangle and the grouping are both part of the question.
     */
    private inline fun forEachWindow(
        operation: DenseMatrixOperation,
        call: DenseCall,
        group: Int,
        action: (rows: Int, width: Int) -> Unit,
    ) {
        val n = call.rows
        when (operation) {
            DenseMatrixOperation.Symv ->
                forEachPanel(n, group) { start, width ->
                    action(if (call.lower) n - (start + width) else start, width)
                }

            DenseMatrixOperation.Syr ->
                forEachPanel(n, group) { start, width ->
                    action(if (call.lower) n - (start + width - 1) else start + 1, width)
                }

            // A triangular traversal's windows are every length below the order, and which end it starts
            // from is the dependency order's, not the triangle's: a solve removes a finished entry from
            // everything still to come, so its windows shrink, and a multiply consumes a column before the
            // columns that would overwrite it, so its windows grow. Transposing swaps the two.
            DenseMatrixOperation.Trsv, DenseMatrixOperation.TrmvTransposed ->
                for (k in n - 1 downTo 0) action(k, 1)

            DenseMatrixOperation.Trmv, DenseMatrixOperation.TrsvTransposed ->
                for (k in 0 until n) action(k, 1)

            // A rectangular operand's columns are all as long as it is tall, whatever the grouping.
            else -> forEachPanel(call.columns, group) { _, width -> action(n, width) }
        }
    }

    private fun panelEntryPoint(work: PanelWork): String = when (work) {
        PanelWork.MultiDot -> "multi-dot"
        PanelWork.ColumnUpdate -> "column-update"
        PanelWork.CoupledDotUpdate -> "coupled-dot-update"
        PanelWork.RankUpdate -> "rank-update"
        PanelWork.SparseRightHandSides -> "sparse-rhs"
    }
}

/** The component name every built-in dense Level 2 and 3 call reports, whatever panels it calls. */
internal const val DENSE_SCHEDULING: String = "portable-dense"
