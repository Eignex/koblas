@file:Suppress("TooManyFunctions") // one description per executed shape, beside the traversal it describes

package com.eignex.koblas.dense

import com.eignex.koblas.vendor.RouteKind

internal class DenseRouteReporter(
    private val vectors: DenseVectorKernels,
    private val panels: DensePanelKernels,
    private val products: DenseProductKernels,
    private val triangles: DenseTriangularKernels,
) {
    fun routeOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        if (call.alpha == 0.0 || call.rows == 0 || call.columns == 0 || call.depth == 0) {
            return route(
                operation,
                RouteKind.NoWork,
                destinationScaling(operation, call, working = false),
                0,
            )
        }
        if (operation in PRODUCT_OPERATIONS) return productRoute(operation, call, OutputTriangle.Full)
        if (operation in TRIANGLE_PRODUCT_OPERATIONS) return productRoute(operation, call, selectedTriangle(call.lower))
        if (operation == DenseMatrixOperation.Symm) return symmetricRoute(operation, call)
        if (operation in TRIANGULAR_MATRIX_OPERATIONS) return triangularMatrixRoute(operation, call)
        val scaling = destinationScaling(operation, call, working = true)
        val work = panelWorkOf(operation)
            ?: return route(
                operation,
                RouteKind.Direct,
                scaling,
                0,
            )
        return panelRoute(operation, call, scaling, work)
    }

    private class Parts {
        val components = ArrayList<String>(6)
        var group: Int = 0
        var composed: Boolean = false
        private var varies: Boolean = false

        fun add(component: String) {
            if (component !in components) components.add(component)
        }

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

    private fun Parts.route(operation: DenseMatrixOperation): DenseMatrixRoute =
        route(operation, if (composed) RouteKind.Composed else RouteKind.Direct, components, group)

    private fun addProductWindow(parts: Parts, call: WindowShape): Boolean {
        if (call.rows <= 0 || call.columns <= 0) return false
        val packs = packsWindow(products, call.rows, call.columns, call.depth, call.selected)
        if (packs) addBlockedWindow(parts, call) else addDirectWindow(parts, call)
        return packs
    }

    private fun addBlockedWindow(parts: Parts, call: WindowShape) {
        // Report panels and tiles in schedule order: right panel, left panel, then tile.
        parts.add("$PRODUCT_PACKING/right-panel")
        parts.add("$PRODUCT_PACKING/left-panel")
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

    private fun addSelectedBlock(
        parts: Parts,
        call: WindowShape,
        row: Int,
        rowCount: Int,
        column: Int,
        columnCount: Int,
        depth: Int,
    ) = forEachSelectedTile(products, row, rowCount, column, columnCount, call.selected) { _, rows, _, columns, whole ->
        for (body in products.implementationsFor(rows, columns, depth)) parts.add("$body/product-block")
        if (!whole) parts.add("$TRIANGLE_SELECTION/triangle-tile")
    }

    private fun addDirectWindow(parts: Parts, call: WindowShape) {
        val work = directPanelWork(call.transposeA)
        val chunk = directColumnBlock(call.rows)
        val strided = call.transposeA && call.rightStride != 1
        val gathers = gathersCoefficients(panels, call.rows, call.depth, strided)
        val contiguous = !strided || gathers
        val group = directPanelGroup(panels, call.rows, call.depth, call.transposeA)
        if (gathers) parts.add("$PRODUCT_PACKING/right-column")
        val entry = panelEntryPoint(work)
        val before = parts.components.size
        for (column in 0 until call.columns) {
            // Selected triangles shorten destination columns, so report their actual panel lengths.
            val rows = selectedRows(column, call.rows, call.selected)
            if (call.transposeA) {
                forEachPanel(rows, group) { _, width ->
                    parts.add("${panels.implementationFor(work, call.depth, width, contiguous)}/$entry")
                }
            } else {
                // A shorter final chunk can reach a different panel body.
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

    private fun productRoute(
        operation: DenseMatrixOperation,
        call: DenseCall,
        selected: OutputTriangle,
    ): DenseMatrixRoute {
        val full = selected == OutputTriangle.Full
        val k = requireNotNull(call.depth) {
            if (full) {
                "a product route needs the call's shared dimension"
            } else {
                "a triangle-selected product route needs the shared dimension"
            }
        }
        val parts = Parts()
        val transposeB = when (operation) {
            DenseMatrixOperation.Syrk, DenseMatrixOperation.Syr2k -> !call.transposeA
            else -> call.transposeB
        }
        val shape = WindowShape(
            call.rows,
            call.columns,
            k,
            call.transposeA,
            transposeB,
            selected,
            rightStride = if (!transposeB) {
                1
            } else if (full) {
                call.rightLeadingDimension ?: call.columns
            } else {
                call.columns
            },
        )
        addProductWindow(parts, shape)
        if (operation == DenseMatrixOperation.Syr2k) parts.composed = true
        return parts.route(
            operation,
        )
    }

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
        return parts.route(
            operation,
        )
    }

    private fun triangularMatrixRoute(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        val order = requireNotNull(call.depth) { "a triangular matrix route needs the triangle's order" }
        val solve = operation == DenseMatrixOperation.Trsm
        val sides = if (call.right) call.rows else call.columns
        val parts = Parts()
        if (call.alpha != 1.0) {
            val length = if (call.leadingDimension == null ||
                call.leadingDimension == call.rows
            ) {
                order * sides
            } else {
                call.rows
            }
            vectors.implementationFor(DenseOperation.Scale, length)?.let { parts.add("$it/scale") }
        }
        val flip = call.transposeA != call.right
        val mLower = call.lower != flip
        val block = if (TRIANGULAR_DIAGONAL_BLOCK < order) TRIANGULAR_DIAGONAL_BLOCK else order
        val rhsStride = if (call.right) 1 else call.leadingDimension ?: order
        val gathers = triangles.gathersRightHandSides(rhsStride, block, sides)
        val group = triangularGroup(triangles, block, sides)
        if (gathers) parts.add("$TRIANGULAR_GATHER/rhs-block")
        val entry = if (solve) "diagonal-solve" else "diagonal-multiply"
        val before = parts.components.size
        forEachTriangularBlock(order, block, mLower, solve) { _, size, _, targetCount, _, sourceCount ->
            forEachPanel(sides, group) { _, lanes ->
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
        // A shorter final diagonal block can reach a different substitution body.
        if (parts.components.size - before > 1) parts.composed = true
        return parts.route(
            operation,
        )
    }

    private fun panelWorkOf(operation: DenseMatrixOperation): PanelWork? = when (operation) {
        DenseMatrixOperation.Gemv, DenseMatrixOperation.Trmv, DenseMatrixOperation.Trsv -> PanelWork.ColumnUpdate

        DenseMatrixOperation.GemvTransposed, DenseMatrixOperation.TrmvTransposed,
        DenseMatrixOperation.TrsvTransposed,
        -> PanelWork.MultiDot

        DenseMatrixOperation.Symv -> PanelWork.CoupledDotUpdate

        DenseMatrixOperation.Ger, DenseMatrixOperation.GerWindow, DenseMatrixOperation.Syr -> PanelWork.RankUpdate

        else -> null
    }

    private fun route(
        operation: DenseMatrixOperation,
        kind: RouteKind,
        components: List<String>,
        group: Int,
    ): DenseMatrixRoute = DenseMatrixRoute(operation, kind, DENSE_SCHEDULING, operation.entryPoint, components, group)

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
        return scaleComponent(vectors, elements)
    }

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
            // Report zero grouping when no panel is scheduled.
            0 -> route(
                operation,
                RouteKind.Direct,
                scaling,
                0,
            )

            1 -> route(
                operation,
                RouteKind.Direct,
                scaling + "${leaves.single()}/$entry",
                group,
            )

            else -> route(
                operation,
                RouteKind.Composed,
                scaling + leaves.map { "$it/$entry" },
                group,
            )
        }
    }

    private inline fun forEachWindow(
        operation: DenseMatrixOperation,
        call: DenseCall,
        group: Int,
        action: (rows: Int, width: Int) -> Unit,
    ) {
        val n = call.rows
        when (operation) {
            DenseMatrixOperation.Symv ->
                forEachTrianglePanel(n, group, call.lower, fromDiagonal = false) { _, width, _, rows ->
                    action(rows, width)
                }

            DenseMatrixOperation.Syr ->
                forEachTrianglePanel(n, group, call.lower, fromDiagonal = true) { _, width, _, rows ->
                    action(rows, width)
                }

            // Solve windows shrink as unknowns finish; multiply windows grow to preserve inputs.
            // Transposition reverses the dependency order.
            DenseMatrixOperation.Trsv, DenseMatrixOperation.TrmvTransposed,
            DenseMatrixOperation.Trmv, DenseMatrixOperation.TrsvTransposed,
            -> {
                val ascending = call.lower == (
                    operation == DenseMatrixOperation.Trsv ||
                        operation == DenseMatrixOperation.TrmvTransposed
                    )
                forEachTriangularColumn(n, call.lower, ascending) { _, _, rows -> action(rows, 1) }
            }

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

private class WindowShape(
    val rows: Int,
    val columns: Int,
    val depth: Int,
    val transposeA: Boolean = false,
    val transposeB: Boolean = false,
    val selected: OutputTriangle = OutputTriangle.Full,
    val rightStride: Int = if (transposeB) columns else 1,
)
