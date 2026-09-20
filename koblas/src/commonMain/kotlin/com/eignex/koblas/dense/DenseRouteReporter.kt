@file:Suppress("TooManyFunctions") // one description per executed shape, beside the traversal it describes

package com.eignex.koblas.dense

import com.eignex.koblas.vendor.RouteKind

/**
 * What a built-in dense Level 2 or 3 call executes, derived from the decisions the call itself makes.
 *
 * Every answer here comes from walking the same traversal the execution walks, with the same grouping and
 * the same block extents, and asking the selected backends what each window of it reaches. That is the
 * difference between a route and a guess: a triangular traversal over four columns cuts windows of three,
 * two, one and nothing, so on a machine whose lane block is four it never reaches a vector body at all, and
 * a description derived from the operation and the order alone would say it did.
 *
 * Nothing here computes. Building a route inspects shapes and allocates a list, which is why it belongs
 * before a timed region and never inside one.
 */
internal class DenseRouteReporter(
    private val vectors: DenseVectorKernels,
    private val panels: DensePanelKernels,
    private val products: DenseProductKernels,
    private val triangles: DenseTriangularKernels,
) {
    /** What a call of [operation] with the facts in [call] executes. */
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
        if (operation in PRODUCT_OPERATIONS) return productRoute(operation, call)
        if (operation in TRIANGLE_PRODUCT_OPERATIONS) return triangleProductRoute(operation, call)
        if (operation == DenseMatrixOperation.Symm) return symmetricRoute(operation, call)
        if (operation in TRIANGULAR_MATRIX_OPERATIONS) return triangularMatrixRoute(operation, call)
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

    /**
     * The components a call reaches, collected in the order it reaches them.
     *
     * A list and two flags rather than a returned description per window, because a structured call is
     * several windows and what a caller wants from them is one route. [composed] records that some unit of
     * work reached a body another did not, which is what makes a route name what may run rather than what
     * does.
     */
    private class Parts {
        val components = ArrayList<String>(6)
        var group: Int = 0
        var composed: Boolean = false
        private var varies: Boolean = false

        fun add(component: String) {
            if (component !in components) components.add(component)
        }

        /**
         * The grouping a window was scheduled with, kept only while every window agrees on one.
         *
         * A structured call is several windows of different shapes, and the backend may recommend a
         * different width for each. One of them would not describe the call, so where they disagree the
         * route reports no grouping at all rather than the one that happened to be asked for last.
         */
        fun useGroup(recommended: Int) {
            if (varies) return
            if (group == 0) {
                group = recommended
            } else if (group != recommended) {
                varies = true
                group = 0
            }
        }
    }

    /**
     * What one product window of these extents executes, appended to [parts].
     *
     * The same question [productWindow] answers when it runs: a window with enough arithmetic to hide a copy
     * of both operands is packed into the backend's tiles, and one without runs as panel work over the
     * operands where they lie. [packLeft] and [packRight] are false where an operand arrived packed, since
     * the copy it would name has already happened.
     */
    private fun addProductWindow(
        parts: Parts,
        call: WindowShape,
        packLeft: Boolean = true,
        packRight: Boolean = true,
    ) {
        if (call.rows <= 0 || call.columns <= 0) return
        if (packsWindow(products, call.rows, call.columns, call.depth, call.selected)) {
            addBlockedWindow(parts, call, packLeft, packRight)
        } else {
            addDirectWindow(parts, call)
        }
    }

    /** The packing and tile bodies a blocked window reaches, taken from walking its own block schedule. */
    private fun addBlockedWindow(parts: Parts, call: WindowShape, packLeft: Boolean, packRight: Boolean) {
        // In the order the schedule reaches them: the right panel belongs to the column and depth block and
        // is packed first, the left panel to the row block inside it, and the tile after both.
        if (packRight) parts.add("$PRODUCT_PACKING/right-panel")
        if (packLeft) parts.add("$PRODUCT_PACKING/left-panel")
        val before = parts.components.size
        forEachProductBlock(
            call.rows,
            call.columns,
            call.depth,
            productBlockRows(products, call.rows),
            productBlockColumns(products, call.columns),
            productBlockDepth(call.depth),
        ) { row, rowCount, column, columnCount, _, depth ->
            if (!outsideTriangle(row, rowCount, column, columnCount, call.selected)) {
                if (insideTriangle(row, rowCount, column, columnCount, call.selected)) {
                    for (body in products.implementationsFor(rowCount, columnCount, depth)) {
                        parts.add("$body/product-block")
                    }
                } else {
                    addSelectedBlock(parts, call, row, rowCount, column, columnCount, depth)
                }
            }
        }
        if (parts.components.size - before > 1) parts.composed = true
    }

    /** The tiles a block straddling the diagonal cuts, and the merge its straddling tiles reach. */
    private fun addSelectedBlock(
        parts: Parts,
        call: WindowShape,
        row: Int,
        rowCount: Int,
        column: Int,
        columnCount: Int,
        depth: Int,
    ) {
        val tileRows = products.tileRows
        val tileColumns = products.tileColumns
        var tileColumn = 0
        while (tileColumn < columnCount) {
            val columns = if (tileColumns < columnCount - tileColumn) tileColumns else columnCount - tileColumn
            var tileRow = 0
            while (tileRow < rowCount) {
                val rows = if (tileRows < rowCount - tileRow) tileRows else rowCount - tileRow
                val at = row + tileRow
                val from = column + tileColumn
                if (!outsideTriangle(at, rows, from, columns, call.selected)) {
                    for (body in products.implementationsFor(rows, columns, depth)) {
                        parts.add("$body/product-block")
                    }
                    if (!insideTriangle(at, rows, from, columns, call.selected)) {
                        parts.add("$TRIANGLE_SELECTION/triangle-tile")
                    }
                }
                tileRow += tileRows
            }
            tileColumn += tileColumns
        }
    }

    /**
     * The panels a window too small or too thin to pack reaches, over the columns its traversal cuts.
     *
     * Which panel that is depends on the left transpose. Whether its shared vector is adjacent is a question
     * only the reducing form has: a column update shares its destination strip, which is always adjacent,
     * while a reduction shares the coefficient column, which a transposed right operand leaves strided by as
     * many entries as the destination has columns. One destination column makes even that adjacent, which is
     * why the stride is worked out rather than read off the flag.
     */
    private fun addDirectWindow(parts: Parts, call: WindowShape) {
        val work = if (call.transposeA) PanelWork.MultiDot else PanelWork.ColumnUpdate
        val chunk = directColumnBlock(call.rows)
        val panelRows = if (call.transposeA) call.depth else chunk
        val panelColumns = if (call.transposeA) call.rows else call.depth
        val strided = call.transposeA && call.rightStride != 1
        val gathers = gathersCoefficients(panels, call.rows, call.depth, strided)
        val contiguous = !strided || gathers
        val group = panels.executionGroup(work, panelRows, panelColumns)
        if (gathers) parts.add("$PRODUCT_PACKING/right-column")
        val entry = panelEntryPoint(work)
        val before = parts.components.size
        for (column in 0 until call.columns) {
            // A selected triangle shortens each destination column, so the panels a column reaches are as
            // long as the triangle is wide there rather than as long as the operand is.
            val rows = selectedRows(column, call.rows, call.selected)
            if (call.transposeA) {
                forEachPanel(rows, group) { _, width ->
                    parts.add("${panels.implementationFor(work, panelRows, width, contiguous)}/$entry")
                }
            } else {
                // The accumulating route cuts each column into chunks of a bounded height, so the last
                // chunk of a column is a shorter panel than the ones before it and need not reach the
                // same body.
                var offset = 0
                while (offset < rows) {
                    val height = if (chunk < rows - offset) chunk else rows - offset
                    forEachPanel(call.depth, group) { _, width ->
                        parts.add("${panels.implementationFor(work, height, width, contiguous)}/$entry")
                    }
                    offset += height
                }
            }
            if (call.selected == OutputTriangle.Full) break
        }
        if (parts.components.size - before > 1) parts.composed = true
        parts.useGroup(group)
    }

    /**
     * What a matrix product executes, which is a question about its extents and how its operands arrived.
     *
     * A call whose operands were packed beforehand always runs in the tiles, and names only the packing it
     * still has to do, which is none where both panels were retained.
     */
    private fun productRoute(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        val k = requireNotNull(call.depth) { "a product route needs the call's shared dimension" }
        val parts = Parts()
        val retained = operation != DenseMatrixOperation.Gemm
        val shape = WindowShape(call.rows, call.columns, k, call.transposeA, call.transposeB, OutputTriangle.Full)
        if (retained) {
            // A packed operand is already in the tiles, so the window is blocked whatever its extents are.
            addBlockedWindow(
                parts,
                shape,
                packLeft = operation == DenseMatrixOperation.GemmPackedRight,
                packRight = operation == DenseMatrixOperation.GemmPackedLeft,
            )
        } else {
            addProductWindow(parts, shape)
        }
        val packed = retained || packsWindow(products, call.rows, call.columns, k, OutputTriangle.Full)
        return route(
            operation,
            if (parts.composed) RouteKind.Composed else RouteKind.Direct,
            parts.components,
            parts.group,
            if (packed) {
                "the product is cut into cache blocks, with beta carried by the first depth block" +
                    if (retained) "; a retained panel is read where it lies rather than packed again" else ""
            } else {
                "too little arithmetic here to pay for packing, so the product runs as panel work down each " +
                    "destination column"
            },
        )
    }

    /**
     * What a product into one triangle of a square destination executes.
     *
     * `gemmt` and `syrk` are one window and `syr2k` is two of the same shape, which is what makes the
     * fused form a comparison rather than an implementation: the two products it composes are ordinary
     * windows, and a fused traversal would have to beat both of them together.
     */
    private fun triangleProductRoute(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        val k = requireNotNull(call.depth) { "a triangle-selected product route needs the shared dimension" }
        val parts = Parts()
        val selected = if (call.lower) OutputTriangle.Lower else OutputTriangle.Upper
        val transposeB = if (operation == DenseMatrixOperation.Gemmt) call.transposeB else !call.transposeA
        val shape = WindowShape(call.rows, call.columns, k, call.transposeA, transposeB, selected)
        addProductWindow(parts, shape)
        val pairs = operation == DenseMatrixOperation.Syr2k
        if (pairs) parts.composed = true
        return route(
            operation,
            if (parts.composed) RouteKind.Composed else RouteKind.Direct,
            parts.components,
            parts.group,
            (
                if (pairs) {
                    "the rank update is the two products it is defined as, composed rather than fused, each "
                } else {
                    "the product is one window, "
                }
                ) +
                "scheduled over the selected triangle only: a block wholly in the other triangle is never " +
                "cut, and one straddling the diagonal reaches the destination a selected entry at a time",
        )
    }

    /**
     * What a symmetric product against a dense block executes.
     *
     * The symmetric operand's diagonal blocks are written out and multiplied as dense squares, and the
     * strips of its stored triangle beside them are multiplied twice, once as stored and once transposed,
     * which is the mirrored half. The windows are walked here exactly as the execution walks them.
     */
    private fun symmetricRoute(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        val order = requireNotNull(call.depth) { "a symmetric product route needs the operand's order" }
        val parts = Parts()
        val block = if (SYMMETRIC_BLOCK < order) SYMMETRIC_BLOCK else order
        for (component in destinationScaling(operation, call, working = true)) parts.add(component)
        parts.add("$SYMMETRIC_EXPANSION/diagonal-block")
        forEachSymmetricBlock(order, block, call.lower) { _, size, _, stripCount ->
            if (call.right) {
                addProductWindow(parts, WindowShape(call.rows, size, size))
                if (stripCount > 0) {
                    addProductWindow(parts, WindowShape(call.rows, size, stripCount))
                    addProductWindow(
                        parts,
                        WindowShape(call.rows, stripCount, size, transposeB = true, rightStride = order),
                    )
                }
            } else {
                addProductWindow(parts, WindowShape(size, call.columns, size))
                if (stripCount > 0) {
                    addProductWindow(parts, WindowShape(stripCount, call.columns, size))
                    addProductWindow(parts, WindowShape(size, call.columns, stripCount, transposeA = true))
                }
            }
        }
        if (order > block) parts.composed = true
        return route(
            operation,
            if (parts.composed) RouteKind.Composed else RouteKind.Direct,
            parts.components,
            parts.group,
            "the symmetric operand is cut into diagonal blocks and the stored strips beside them; a strip " +
                "is multiplied as stored and again transposed, which is the half that is not stored, and " +
                "only a diagonal block is copied",
        )
    }

    /**
     * What a triangular matrix multiply or solve executes.
     *
     * The diagonal blocks reach the selected substitution backend and the products between them are
     * ordinary windows of shared product scheduling. Both are walked here over the same blocks the call
     * cuts, so a call whose order fits one block names no product at all and says so.
     */
    private fun triangularMatrixRoute(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        val order = requireNotNull(call.depth) { "a triangular matrix route needs the triangle's order" }
        val solve = operation == DenseMatrixOperation.Trsm
        val sides = if (call.right) call.rows else call.columns
        val parts = Parts()
        if (call.alpha != 1.0) {
            vectors.implementationFor(DenseOperation.Scale, order * sides)?.let { parts.add("$it/scale") }
        }
        val flip = call.transposeA != call.right
        val mLower = call.lower != flip
        val block = if (TRIANGULAR_DIAGONAL_BLOCK < order) TRIANGULAR_DIAGONAL_BLOCK else order
        val rhsStride = if (call.right) 1 else order
        val gathers = triangles.gathersRightHandSides(rhsStride, block, sides)
        val group = triangularGroup(triangles, block, sides)
        if (gathers) parts.add("$TRIANGULAR_GATHER/rhs-block")
        val entry = if (solve) "diagonal-solve" else "diagonal-multiply"
        val before = parts.components.size
        forEachTriangularBlock(order, block, mLower, solve) { _, size, _, targetCount, _, sourceCount ->
            forEachRightHandSideGroup(sides, group) { _, lanes ->
                for (body in triangles.implementationsFor(size, lanes, gathers || rhsStride == 1)) {
                    parts.add("$body/$entry")
                }
            }
            if (targetCount > 0 && sourceCount > 0) {
                if (call.right) {
                    addProductWindow(
                        parts,
                        WindowShape(sides, targetCount, sourceCount, transposeB = !flip, rightStride = order),
                    )
                } else {
                    addProductWindow(parts, WindowShape(targetCount, sides, sourceCount, transposeA = flip))
                }
            }
        }
        // A call whose order is not a multiple of the block ends on a shorter one, and the substitution that
        // block reaches need not be the one the full blocks reached.
        if (parts.components.size - before > 1) parts.composed = true
        return route(
            operation,
            if (parts.composed) RouteKind.Composed else RouteKind.Direct,
            parts.components,
            parts.group,
            "the dependency chain is cut into diagonal blocks of $block, each substituted over $group " +
                "right-hand sides at a time" +
                (if (gathers) " in a gathered copy of them" else " where they lie") +
                if (order > block) {
                    ", and the product between one block and the steps it still reaches is an ordinary window"
                } else {
                    ", and the order fits one block, so no product runs between them"
                },
        )
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
     * kernel like the rest.
     *
     * Most of Level 3 folds the multiplier into the product block or the panel that writes each output, and
     * scales what is left of its selected region with a loop of its own, which is no kernel and is named as
     * nothing. A symmetric product cannot: every entry of its destination is accumulated into by several of
     * the windows the symmetric operand is cut into and none of them owns it, so the multiplier is a pass
     * of its own over the whole buffer, it is the Level 1 kernel that makes it, and the route says so.
     */
    private fun destinationScaling(operation: DenseMatrixOperation, call: DenseCall, working: Boolean): List<String> {
        val scales = when (operation) {
            DenseMatrixOperation.Gemv, DenseMatrixOperation.Symv, DenseMatrixOperation.Symm -> true
            DenseMatrixOperation.GemvTransposed -> !working
            else -> false
        }
        val elements = when (operation) {
            DenseMatrixOperation.GemvTransposed -> call.columns
            DenseMatrixOperation.Symm -> call.rows * call.columns
            else -> call.rows
        }
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
     * Written out beside the traversals rather than derived from them, so that a route and a call can be
     * compared with each other. The lengths are what a triangle's storage leaves at each group, which is
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
        PanelWork.SparseRightHandSideReduction -> "sparse-rhs-reduction"
    }
}

/**
 * One product window as the route sees it: extents, both transposes and which triangle it may write.
 *
 * The same six numbers [productWindow] takes its decisions from, without the operands it takes them for.
 */
private class WindowShape(
    val rows: Int,
    val columns: Int,
    val depth: Int,
    val transposeA: Boolean = false,
    val transposeB: Boolean = false,
    val selected: OutputTriangle = OutputTriangle.Full,
    /**
     * How far apart the coefficients of one destination column are, which decides whether a reduction can
     * vectorise over them and so whether the direct route gathers.
     *
     * Adjacent for an untransposed right operand. For a transposed one it is that operand's stored leading
     * dimension, which is the window's own columns where the window is the whole matrix and something else
     * where it is a strip of one, so a caller handing over a strip states it.
     */
    val rightStride: Int = if (transposeB) columns else 1,
)
