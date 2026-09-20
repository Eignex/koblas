package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.dense.PanelWork
import com.eignex.koblas.vendor.RouteKind
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * That a sparse route names the panel bodies the call actually ran.
 *
 * A route claims something about the runs of stored entries a call hands over, and checking that means
 * recording what was handed over rather than reasoning from a column's length: a triangular routine passes
 * the strictly triangular part of a column, a symmetric one passes the selected part in one coupled pass,
 * a product passes the whole column, and a matrix can store nothing that any of them selects. The backend
 * below writes down every run it is given and computes with the portable bodies, so the comparison is
 * between the route and the execution.
 */
class SparsePanelRouteTest {

    @Test
    fun `a product names the body its panels reached`() {
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val recorder = RecordingRhsPanels()
                val engine = engineWith(recorder)
                val rng = Random(20261010)
                val m = ORDER
                val k = DEPTH
                val a = spread(if (transposeA) k else m, if (transposeA) m else k, rng)
                val b = if (transposeB) dense(SIDES, k, rng) else dense(k, SIDES, rng)
                val c = dense(m, SIDES, rng)

                engine.gemm(0.875, a, transposeA, b, transposeB, -0.25, c, right = false, workspace = Workspace())
                val route = engine.matrixRouteOf(
                    SparseMatrixOperation.GemmDense,
                    SparseCall(
                        a,
                        0.875,
                        -0.25,
                        destinationElements = c.values.size,
                        depth = k,
                        rightHandSides = SIDES,
                        transposeSparse = transposeA,
                        transposeDense = transposeB,
                    ),
                )
                assertEquals(
                    recorder.bodies(),
                    panelBodies(route),
                    "transposeA=$transposeA transposeB=$transposeB over runs ${recorder.runs}",
                )
            }
        }
    }

    @Test
    fun `a symmetric product names the body its coupled runs reached`() {
        for (lower in booleanArrayOf(true, false)) {
            val recorder = RecordingRhsPanels()
            val engine = engineWith(recorder)
            val rng = Random(20261011)
            val order = ORDER
            val a = spread(order, order, rng)
            val b = dense(order, SIDES, rng)
            val c = dense(order, SIDES, rng)

            engine.symm(0.875, a, b, -0.25, c, lower, right = false, workspace = Workspace())
            val route = engine.matrixRouteOf(
                SparseMatrixOperation.SymmLeft,
                SparseCall(
                    a,
                    0.875,
                    -0.25,
                    destinationElements = c.values.size,
                    depth = order,
                    rightHandSides = SIDES,
                    lower = lower,
                ),
            )
            assertEquals(recorder.bodies(), panelBodies(route), "lower=$lower over runs ${recorder.runs}")
        }
    }

    @Test
    fun `a triangular block names the bodies its strictly triangular runs reached`() {
        for (solve in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                val recorder = RecordingRhsPanels()
                val engine = engineWith(recorder)
                val rng = Random(20261012)
                val order = ORDER
                val t = triangle(order, rng)
                val b = dense(order, SIDES, rng)

                if (solve) {
                    engine.trsm(t, b, true, transpose, false, false, 0.875, Workspace())
                } else {
                    engine.trmm(t, b, true, transpose, false, false, 0.875, Workspace())
                }
                val route = engine.matrixRouteOf(
                    if (solve) SparseMatrixOperation.TrsmLeft else SparseMatrixOperation.TrmmLeft,
                    SparseCall(
                        t,
                        0.875,
                        destinationElements = b.values.size,
                        rightHandSides = SIDES,
                        transposeSparse = transpose,
                        lower = true,
                    ),
                )
                assertEquals(
                    recorder.bodies(),
                    panelBodies(route),
                    "solve=$solve transpose=$transpose over runs ${recorder.runs}",
                )
            }
        }
    }

    /**
     * A triangle storing only its diagonal hands over nothing, so no panel body runs and none is named.
     *
     * Naming one here would be the whole failure this check exists for: the columns are not empty, and a
     * route that read their lengths rather than their selected runs would report a body, on a vector
     * backend a vector one, for arithmetic that never happened.
     */
    @Test
    fun `a diagonal only triangle reaches no panel at all`() {
        val recorder = RecordingRhsPanels()
        val engine = engineWith(recorder)
        val order = 6
        val t = SparseMatrix.ofColumns(order, order, List(order) { j -> listOf(j to 2.0 + j) })
        val b = dense(order, SIDES, Random(20261013))

        engine.trsm(t, b, true, false, false, false, 0.875, Workspace())
        val route = engine.matrixRouteOf(
            SparseMatrixOperation.TrsmLeft,
            SparseCall(t, 0.875, destinationElements = b.values.size, rightHandSides = SIDES, lower = true),
        )

        assertTrue(recorder.runs.isEmpty(), "a diagonal-only triangle handed over ${recorder.runs}")
        assertEquals(emptyList(), panelBodies(route), "a diagonal-only triangle named a panel body")
        assertEquals(RouteKind.Direct, route.kind)
    }

    /** A matrix storing only the triangle this call does not select is the same question from the other side. */
    @Test
    fun `a matrix storing only the unselected triangle reaches no panel at all`() {
        val recorder = RecordingRhsPanels()
        val engine = engineWith(recorder)
        val order = 6
        // Strictly upper, and the call selects the lower triangle, so nothing it stores is ever handed over.
        val a = SparseMatrix.ofColumns(order, order, List(order) { j -> (0 until j).map { i -> i to 1.0 + i } })
        val b = dense(order, SIDES, Random(20261014))
        val c = dense(order, SIDES, Random(20261015))

        engine.symm(0.875, a, b, -0.25, c, lower = true, right = false, workspace = Workspace())
        val route = engine.matrixRouteOf(
            SparseMatrixOperation.SymmLeft,
            SparseCall(
                a,
                0.875,
                -0.25,
                destinationElements = c.values.size,
                depth = order,
                rightHandSides = SIDES,
                lower = true,
            ),
        )

        assertTrue(recorder.runs.isEmpty(), "an unselected triangle handed over ${recorder.runs}")
        assertEquals(emptyList(), panelBodies(route), "an unselected triangle named a panel body")
    }

    /**
     * A last group narrower than the recommendation, and a backend whose body depends on the width.
     *
     * The route asks about both widths a call cuts, so a short last group that reaches a different body is
     * named beside the full ones and the call is published as the composition it is.
     */
    @Test
    fun `a short last group of right hand sides is named beside the full ones`() {
        val recorder = RecordingRhsPanels(group = 4, minimumRows = 4)
        val engine = engineWith(recorder)
        val rng = Random(20261016)
        val m = ORDER
        val k = DEPTH
        val sides = 6
        val a = spread(m, k, rng)
        val b = dense(k, sides, rng)
        val c = dense(m, sides, rng)

        engine.gemm(0.875, a, false, b, false, -0.25, c, right = false, workspace = Workspace())
        val route = engine.matrixRouteOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(
                a,
                0.875,
                -0.25,
                destinationElements = c.values.size,
                depth = k,
                rightHandSides = sides,
            ),
        )

        assertEquals(setOf(4, 2), recorder.runs.map { it.rows }.toSet(), "the groups cut were ${recorder.runs}")
        assertEquals(recorder.bodies(), panelBodies(route))
        assertEquals(2, panelBodies(route).size, "the short group reached no body of its own")
        assertEquals(RouteKind.Composed, route.kind)
    }

    /**
     * The same question for a traversal that runs the other way.
     *
     * A triangular multiply walks its columns from the far end where a solve walks them from the near one,
     * so a route that always scanned forward would name the same two bodies in the reverse of the order
     * they ran. First-seen order is what a component list promises, so the check is the list itself.
     */
    @Test
    fun `a count sensitive backend names the bodies of a reversed traversal in the order they ran`() {
        for (solve in booleanArrayOf(true, false)) {
            for (lower in booleanArrayOf(true, false)) {
                val recorder = RecordingRhsPanels(entryThreshold = 3)
                val engine = engineWith(recorder)
                val order = ORDER
                val t = straddling(order, lower)
                val b = dense(order, SIDES, Random(20261019))

                if (solve) {
                    engine.trsm(t, b, lower, false, false, false, 0.875, Workspace())
                } else {
                    engine.trmm(t, b, lower, false, false, false, 0.875, Workspace())
                }
                val route = engine.matrixRouteOf(
                    if (solve) SparseMatrixOperation.TrsmLeft else SparseMatrixOperation.TrmmLeft,
                    SparseCall(
                        t,
                        0.875,
                        destinationElements = b.values.size,
                        rightHandSides = SIDES,
                        lower = lower,
                    ),
                )

                val context = "solve=$solve lower=$lower over ${recorder.runs}"
                assertTrue(recorder.bodies().size > 1, "the fixture reached one body: $context")
                assertEquals(recorder.bodies(), panelBodies(route), context)
            }
        }
    }

    /**
     * A backend whose choice of body depends on how many stored entries a run holds.
     *
     * Neither shipped backend asks that question today, and a route that assumed so would be reporting the
     * whole column's length for a run that is shorter. The fixture's columns straddle the threshold.
     */
    @Test
    fun `a count sensitive backend names every body its runs reached`() {
        val recorder = RecordingRhsPanels(entryThreshold = 3)
        val engine = engineWith(recorder)
        val order = 24
        val columns = List(order) { j ->
            // Short columns and long ones alternate, so the runs straddle the backend's threshold.
            if (j % 2 == 0) {
                listOf(j to 2.0 + j) + (j + 1 until order).map { it to 1.0 }
            } else {
                listOf(j to 2.0 + j) + (minOf(j + 1, order - 1) until order).map { it to 0.5 }
            }
        }
        val t = SparseMatrix.ofColumns(
            order,
            order,
            columns.map { it.distinctBy { e -> e.first }.sortedBy { e -> e.first } },
        )
        val b = dense(order, SIDES, Random(20261017))

        engine.trsm(t, b, true, false, false, false, 0.875, Workspace())
        val route = engine.matrixRouteOf(
            SparseMatrixOperation.TrsmLeft,
            SparseCall(t, 0.875, destinationElements = b.values.size, rightHandSides = SIDES, lower = true),
        )

        assertTrue(recorder.bodies().size > 1, "the fixture reached one body: ${recorder.runs}")
        assertEquals(recorder.bodies(), panelBodies(route))
    }

    /**
     * A single right-hand side is the panel's arithmetic written out, and the route says so.
     *
     * The answer is the same either way, so what this checks is the attribution: no panel body ran, and
     * naming one would be reporting a seam the traversal did not use.
     */
    @Test
    fun `a single right hand side names no panel body`() {
        val recorder = RecordingRhsPanels()
        val engine = engineWith(recorder)
        val rng = Random(20261021)
        val a = spread(ORDER, DEPTH, rng)
        val b = dense(DEPTH, 1, rng)
        val c = dense(ORDER, 1, rng)

        engine.gemm(0.875, a, false, b, false, -0.25, c, right = false, workspace = Workspace())
        val route = engine.matrixRouteOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(a, 0.875, -0.25, destinationElements = ORDER, depth = DEPTH, rightHandSides = 1),
        )

        assertTrue(recorder.runs.isEmpty(), "a single right-hand side reached a panel: ${recorder.runs}")
        assertEquals(emptyList(), panelBodies(route), route.toString())
        assertTrue(route.reason.orEmpty().contains("written out"), route.toString())
        assertEquals(1, route.executionGroup)
    }

    /**
     * The same answer whichever path a call takes, to the rounding a grouped sum is allowed.
     *
     * A panel sums a column's entries in its own grouping and the written-out column sums them as it reaches
     * them, which is the reassociation every grouped product here may make; what may not differ is which
     * products were formed or where they landed.
     */
    @Test
    fun `a single right hand side agrees with the panel over the same operand`() {
        val engine = engineWith(AdjacentPreferringPanels())
        val rng = Random(20261022)
        val a = filled(ORDER, rng)
        val wide = dense(ORDER, 3, rng)
        val start = dense(ORDER, 3, rng)

        val batched = DenseMatrix.wrap(ORDER, 3, start.values.copyOf())
        engine.symm(0.875, a, wide, -0.25, batched, lower = true, right = false, workspace = Workspace())
        for (side in 0 until 3) {
            val column = DenseMatrix.wrap(ORDER, 1, wide.values.copyOfRange(side * ORDER, (side + 1) * ORDER))
            val single = DenseMatrix.wrap(
                ORDER,
                1,
                start.values.copyOfRange(side * ORDER, (side + 1) * ORDER),
            )
            engine.symm(0.875, a, column, -0.25, single, lower = true, right = false, workspace = Workspace())
            for (row in 0 until ORDER) {
                assertClose(
                    batched.values[row + side * ORDER],
                    single.values[row],
                    "side $side row $row between the panel and the written-out column",
                )
            }
        }
    }

    /**
     * A triangular routine writes out one right-hand side too, and its route says so.
     *
     * What the panel keeps for a group, the liveness of each right-hand side, is one branch here, so there
     * is nothing left for the seam to carry. Both directions are checked because they are different leaves:
     * one spreads a finished pivot and the other reduces into it.
     */
    @Test
    fun `a triangular call writes out one right hand side`() {
        for (solve in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                val recorder = RecordingRhsPanels()
                val engine = engineWith(recorder)
                val rng = Random(20261023)
                val t = triangle(ORDER, rng)
                val b = dense(ORDER, 1, rng)

                if (solve) {
                    engine.trsm(t, b, true, transpose, false, false, 0.875, Workspace())
                } else {
                    engine.trmm(t, b, true, transpose, false, false, 0.875, Workspace())
                }
                val route = engine.matrixRouteOf(
                    if (solve) SparseMatrixOperation.TrsmLeft else SparseMatrixOperation.TrmmLeft,
                    SparseCall(
                        t,
                        0.875,
                        destinationElements = ORDER,
                        rightHandSides = 1,
                        transposeSparse = transpose,
                        lower = true,
                    ),
                )

                val context = "solve=$solve transpose=$transpose"
                assertTrue(recorder.runs.isEmpty(), "$context reached a panel: ${recorder.runs}")
                assertEquals(emptyList(), panelBodies(route), "$context: $route")
                assertTrue(route.reason.orEmpty().contains("written out"), "$context: $route")
                assertEquals(1, route.executionGroup, context)
            }
        }
    }

    /**
     * A triangular column written out agrees with the same column through a panel, in both directions.
     *
     * The written-out leaf carries the substitution's own rules, the skip and the raw-value liveness, as
     * well as its arithmetic, so what it has to match is the panel over right-hand sides that are all live,
     * which is where the panel and the traversal are specified to agree.
     */
    @Test
    fun `a single right hand side agrees with the triangular panel`() {
        val engine = engineWith(AdjacentPreferringPanels())
        val rng = Random(20261101)
        val t = triangle(ORDER, rng)
        val sides = 3

        for (solve in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                val start = dense(ORDER, sides, rng)
                val batched = DenseMatrix.wrap(ORDER, sides, start.values.copyOf())
                if (solve) {
                    engine.trsm(t, batched, true, transpose, false, false, 0.875, Workspace())
                } else {
                    engine.trmm(t, batched, true, transpose, false, false, 0.875, Workspace())
                }
                for (side in 0 until sides) {
                    val single = DenseMatrix.wrap(
                        ORDER,
                        1,
                        start.values.copyOfRange(side * ORDER, (side + 1) * ORDER),
                    )
                    if (solve) {
                        engine.trsm(t, single, true, transpose, false, false, 0.875, Workspace())
                    } else {
                        engine.trmm(t, single, true, transpose, false, false, 0.875, Workspace())
                    }
                    for (row in 0 until ORDER) {
                        assertClose(
                            batched.values[row + side * ORDER],
                            single.values[row],
                            "solve=$solve transpose=$transpose side $side row $row",
                        )
                    }
                }
            }
        }
    }

    /**
     * A product whose last group holds one right-hand side runs that group written out and the full ones
     * through panels, and the route names both halves.
     *
     * The tail is the case a bypass written as a property of the whole call would get wrong: the call is
     * neither all panel nor all traversal.
     */
    @Test
    fun `a product with a last group of one names the panels and the written out tail`() {
        for (symmetric in booleanArrayOf(false, true)) {
            val recorder = RecordingRhsPanels(group = 4, minimumRows = 1)
            val engine = engineWith(recorder)
            val rng = Random(20261024)
            val sides = 5
            val a = if (symmetric) filled(ORDER, rng) else spread(ORDER, DEPTH, rng)
            val b = dense(if (symmetric) ORDER else DEPTH, sides, rng)
            val c = dense(ORDER, sides, rng)

            if (symmetric) {
                engine.symm(0.875, a, b, -0.25, c, lower = true, right = false, workspace = Workspace())
            } else {
                engine.gemm(0.875, a, false, b, false, -0.25, c, right = false, workspace = Workspace())
            }
            val route = engine.matrixRouteOf(
                if (symmetric) SparseMatrixOperation.SymmLeft else SparseMatrixOperation.GemmDense,
                SparseCall(
                    a,
                    0.875,
                    -0.25,
                    destinationElements = ORDER * sides,
                    depth = if (symmetric) ORDER else DEPTH,
                    rightHandSides = sides,
                    lower = true,
                ),
            )

            val context = "symmetric=$symmetric over ${recorder.runs.map { it.rows }.distinct()}"
            assertEquals(4, route.executionGroup, context)
            assertEquals(1, route.executionTail, context)
            assertTrue(recorder.runs.none { it.rows == 1 }, "$context: a tail of one reached a panel")
            assertEquals(recorder.bodies(), panelBodies(route), context)
            assertEquals(RouteKind.Composed, route.kind, context)
            assertTrue(route.reason.orEmpty().contains("written out"), route.toString())
        }
    }

    /**
     * The same tail for a triangular routine, in both directions, with the liveness rule named beside it.
     *
     * The scattering direction's panel is not unconditional the way a product's is: a group whose
     * right-hand sides are all live reaches the leaf and a group holding a zero keeps the written-out loop
     * that skips it, so the route has to say both or a reader would take the named body for one that always
     * runs. The gathering direction has no such rule, and its route may not claim one.
     */
    @Test
    fun `a triangular call with a last group of one names the panels the tail and the liveness rule`() {
        for (solve in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                val recorder = RecordingRhsPanels(group = 4, minimumRows = 1)
                val engine = engineWith(recorder)
                val rng = Random(20261102)
                val sides = 5
                val t = triangle(ORDER, rng)
                val b = dense(ORDER, sides, rng)

                if (solve) {
                    engine.trsm(t, b, true, transpose, false, false, 0.875, Workspace())
                } else {
                    engine.trmm(t, b, true, transpose, false, false, 0.875, Workspace())
                }
                val route = engine.matrixRouteOf(
                    if (solve) SparseMatrixOperation.TrsmLeft else SparseMatrixOperation.TrmmLeft,
                    SparseCall(
                        t,
                        0.875,
                        destinationElements = ORDER * sides,
                        rightHandSides = sides,
                        transposeSparse = transpose,
                        lower = true,
                    ),
                )

                val context = "solve=$solve transpose=$transpose over ${recorder.runs.map { it.rows }}"
                assertEquals(4, route.executionGroup, context)
                assertEquals(1, route.executionTail, context)
                assertTrue(recorder.runs.none { it.rows == 1 }, "$context: a tail of one reached a panel")
                assertEquals(recorder.bodies(), panelBodies(route), context)
                assertEquals(RouteKind.Composed, route.kind, context)
                assertTrue(route.reason.orEmpty().contains("written out"), "$context: $route")
                assertEquals(!transpose, route.reason.orEmpty().contains("live"), "$context: $route")
            }
        }
    }

    /**
     * A symmetric column is one coupled pass over its selected run, and the route names one body for it.
     *
     * The fixture is what separates that from the two-pass shape it replaced: one column stores every row,
     * so its selected run and its strictly triangular run are different lengths, and a backend whose body
     * depends on the length answers differently for the two. A route that still enumerated both would name
     * a second body for a call that is never made.
     */
    @Test
    fun `a symmetric column names one body for its coupled run`() {
        val recorder = RecordingRhsPanels(group = 4, minimumRows = 1, entryThreshold = 3)
        val engine = engineWith(recorder)
        val order = 3
        val a = SparseMatrix.ofColumns(
            order,
            order,
            listOf(listOf(0 to 2.0, 1 to 1.0, 2 to 1.0), emptyList(), emptyList()),
        )
        val sides = 4
        val b = dense(order, sides, Random(20261025))
        val c = dense(order, sides, Random(20261026))

        engine.symm(0.875, a, b, -0.25, c, lower = true, right = false, workspace = Workspace())
        val route = engine.matrixRouteOf(
            SparseMatrixOperation.SymmLeft,
            SparseCall(
                a,
                0.875,
                -0.25,
                destinationElements = order * sides,
                depth = order,
                rightHandSides = sides,
                lower = true,
            ),
        )

        assertEquals(listOf(3), recorder.runs.map { it.columns }, "the runs a symmetric column handed over")
        assertEquals(recorder.bodies(), panelBodies(route), route.toString())
        assertEquals(1, panelBodies(route).size, "a second body was named for a call that never happened")
    }

    /** What a staged call copies is named for what it is: a source is read, a destination is written back. */
    @Test
    fun `a staged call names the copies it makes`() {
        val engine = engineWith(AdjacentPreferringPanels())
        val rng = Random(20261018)
        val m = ORDER
        val k = DEPTH
        val a = spread(m, k, rng)

        val scatter = engine.matrixRouteOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(a, 0.875, -0.25, destinationElements = m * SIDES, depth = k, rightHandSides = SIDES),
        )
        assertEquals(listOf("$SPARSE_STAGING/rhs-destination"), staging(scatter), "the scattered half")

        val transposed = spread(k, m, rng)
        val gather = engine.matrixRouteOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(
                transposed,
                0.875,
                -0.25,
                destinationElements = m * SIDES,
                depth = k,
                rightHandSides = SIDES,
                transposeSparse = true,
            ),
        )
        assertEquals(listOf("$SPARSE_STAGING/rhs-source"), staging(gather), "the gathered half")

        val symmetric = engine.matrixRouteOf(
            SparseMatrixOperation.SymmLeft,
            SparseCall(
                filled(m, rng),
                0.875,
                -0.25,
                destinationElements = m * SIDES,
                depth = m,
                rightHandSides = SIDES,
            ),
        )
        assertEquals(
            listOf("$SPARSE_STAGING/rhs-source", "$SPARSE_STAGING/rhs-destination"),
            staging(symmetric),
            "the symmetric product copies both blocks",
        )
    }

    // A body's own name may hold a slash, so the leaf is what follows the last one and the body is the rest.
    private fun panelBodies(route: SparseMatrixRoute): List<String> =
        route.components.filter { it.substringAfterLast('/') in PANEL_LEAVES }.map { it.substringBeforeLast('/') }

    private fun staging(route: SparseMatrixRoute): List<String> =
        route.components.filter { it.startsWith(SPARSE_STAGING) }

    private fun dense(rows: Int, cols: Int, rng: Random): DenseMatrix =
        DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

    /** Columns of differing length, including one with nothing stored at all. */
    private fun spread(rows: Int, cols: Int, rng: Random): SparseMatrix = SparseMatrix.ofColumns(
        rows,
        cols,
        List(cols) { j ->
            if (j == 1) emptyList() else (0 until rows).map { it to rng.nextDouble() }
        },
    )

    /**
     * A triangle whose strictly triangular runs straddle a count-sensitive backend's threshold, with the
     * short runs at the end the traversal reaches last.
     *
     * The first half of the columns hold long runs and the second half short ones, in the order a forward
     * traversal meets them, so a solve and a multiply over the same matrix reach the two bodies in opposite
     * orders.
     */
    private fun straddling(order: Int, lower: Boolean): SparseMatrix = SparseMatrix.ofColumns(
        order,
        order,
        List(order) { j ->
            val rows = if (lower) (j until order).toList() else (0..j).toList()
            val long = if (lower) j < order / 2 else j >= order / 2
            val kept = if (long) rows else rows.filter { it == j || it == rows.first() || it == rows.last() }
            kept.distinct().sorted().map { i -> i to if (i == j) 2.0 + order else 0.5 }
        },
    )

    /** Every position stored, which is what a symmetric product needs before its three copies pay for it. */
    private fun filled(order: Int, rng: Random): SparseMatrix = SparseMatrix.ofColumns(
        order,
        order,
        List(order) { j -> (0 until order).map { i -> i to rng.nextDouble() } },
    )

    private fun triangle(order: Int, rng: Random): SparseMatrix = SparseMatrix.ofColumns(
        order,
        order,
        List(order) { j ->
            (j until order).filter { it == j || (it + j) % 3 != 0 }
                .map { i -> i to if (i == j) 2.0 + abs(rng.nextDouble()) else rng.nextDouble(-1.0, 1.0) }
        },
    )

    private companion object {
        const val SIDES = 5

        /** Extents the staging rule admits, as in the staged panel tests beside these. */
        const val ORDER = 24
        const val DEPTH = 16
        val PANEL_LEAVES = setOf("sparse-rhs-gather", "sparse-rhs-scatter", "sparse-rhs-mirror")
    }
}

/**
 * One run of stored entries handed to a panel: how many right-hand sides, how many entries, and which of
 * the two sparse panel shapes it was, since a backend may answer differently for each.
 */
internal data class RecordedRun(val rows: Int, val columns: Int, val contiguous: Boolean, val work: PanelWork)

/** A backend that writes down every run it is handed and computes with the portable bodies. */
internal class RecordingRhsPanels(group: Int = 3, minimumRows: Int = 1, private val entryThreshold: Int? = null) :
    AdjacentPreferringPanels(group, minimumRows, entryThreshold) {
    val runs: MutableList<RecordedRun> = ArrayList()

    /** The distinct bodies the recorded runs reached, in first-seen order, as a route lists them. */
    fun bodies(): List<String> = runs
        .map { implementationFor(it.work, it.rows, it.columns, it.contiguous) }
        .distinct()

    override fun indexedColumnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
    ) {
        if (rows > 0 && columns > 0) {
            runs.add(RecordedRun(rows, columns, rowStride == 1, PanelWork.SparseRightHandSideReduction))
        }
        super.indexedColumnUpdate(
            alpha, a, aOffset, rowStride, indexStride, indices, values, fromIndex, columns, rows, y, yOffset,
        )
    }

    override fun indexedRankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        x: DoubleArray,
        xOffset: Int,
    ) {
        if (rows > 0 && columns > 0) {
            runs.add(RecordedRun(rows, columns, rowStride == 1, PanelWork.SparseRightHandSides))
        }
        super.indexedRankUpdate(
            alpha, a, aOffset, rowStride, indexStride, indices, values, fromIndex, columns, rows, x, xOffset,
        )
    }

    override fun indexedCoupledUpdate(
        alpha: Double,
        a: DoubleArray,
        b: DoubleArray,
        offset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        pivot: Int,
        sums: DoubleArray,
        sumOffset: Int,
        excluded: Int,
    ) {
        if (rows > 0 && columns > 0) {
            runs.add(RecordedRun(rows, columns, rowStride == 1, PanelWork.SparseRightHandSides))
        }
        super.indexedCoupledUpdate(
            alpha, a, b, offset, rowStride, indexStride, indices, values, fromIndex, columns, rows,
            pivot, sums, sumOffset, excluded,
        )
    }
}
