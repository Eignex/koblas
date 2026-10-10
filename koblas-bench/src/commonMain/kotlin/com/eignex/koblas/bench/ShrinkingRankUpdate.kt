package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.PanelWork
import com.eignex.koblas.vendor.RouteKind

/** Reused, disjoint inputs for a sequence of rank updates with successively shorter trailing supports. */
internal class ShrinkingRankUpdate(val order: Int) {
    val matrix: DenseMatrix = Fixtures.matrix(order, order, 1).also { a ->
        // The primary order-six fixture has thirty-one nonzeros, matching the measured basis density.
        for (j in 1 until order) a[0, j] = 0.0
    }
    private val initial = matrix.values.copyOf()
    val x: Array<DoubleArray> = Array(order - 1) { step ->
        DoubleArray(order) { i ->
            if (i <= step) 0.0 else (if (i % 2 == 0) -1.0 else 1.0) * (i % 7 + 1) / 16.0
        }
    }
    val y: Array<DoubleArray> = Array(order - 1) { step ->
        DoubleArray(order) { j ->
            if (j <= step) 0.0 else (if (j % 3 == 0) -1.0 else 1.0) * (j % 5 + 1) / 16.0
        }
    }

    /** Restores the initial matrix outside timing, after checking the entire update sequence. */
    fun verify(what: String, run: () -> Double) {
        var expected = initial.copyOf()
        for (step in x.indices) {
            expected = DenseReference.ger(-1.0, x[step], y[step], DenseMatrix.wrap(order, order, expected))
        }
        initial.copyInto(matrix.values)
        run()
        DenseReference.check(expected, matrix.values, what)
        initial.copyInto(matrix.values)
    }
}

/** Full-buffer GER batches, safe windows and their raw trailing panels. */
internal fun shrinkingRankUpdateWork(case: BenchCase, engine: KoblasEngine): CaseWork {
    val fixture = ShrinkingRankUpdate(case.dimension(0))
    val n = fixture.order
    val windows = case.operation == "panel-rankupdate-shrinking"
    val publicWindows = case.operation == "ger-window-shrinking"
    val panels = engine.panelKernels
    val run: () -> Double = if (windows) {
        {
            for (step in fixture.x.indices) {
                val start = step + 1
                val extent = n - start
                panels.rankUpdate(
                    -1.0, fixture.matrix.values, start + start * n, n,
                    fixture.x[step], start, 1, extent, extent, fixture.y[step], start, 1,
                )
            }
            fixture.matrix.values.last()
        }
    } else if (publicWindows) {
        {
            for (step in fixture.x.indices) {
                val start = step + 1
                val extent = n - start
                engine.ger(
                    -1.0, fixture.x[step], fixture.y[step], fixture.matrix,
                    start, start, extent, extent, start, start,
                )
            }
            fixture.matrix.values.last()
        }
    } else {
        {
            for (step in fixture.x.indices) engine.ger(-1.0, fixture.x[step], fixture.y[step], fixture.matrix)
            fixture.matrix.values.last()
        }
    }
    fixture.verify(case.id, run)
    if (publicWindows) {
        val routes = (n - 1 downTo 1).map { extent ->
            engine.routeOf(DenseMatrixOperation.GerWindow, DenseCall(extent, extent, alpha = -1.0))
        }
        val bodies = routes.flatMap { it.components }.distinct()
        val groups = routes.map { it.executionGroup }.distinct()
        val kind = if (bodies.size == 1) RouteKind.Direct else RouteKind.Composed
        return CaseWork(
            kind.name.lowercase(), "arithmetic", run, result = fixture.matrix.values,
            kernel = "${routes.first().scheduling}+${bodies.joinToString("+")}/ger-window@${groups.joinToString("+")}",
        )
    }
    if (!windows) {
        val route = engine.routeOf(DenseMatrixOperation.Ger, DenseCall(n, n, alpha = -1.0))
        return CaseWork(
            route.kind.name.lowercase(), "arithmetic", run,
            result = fixture.matrix.values, kernel = matrixKernel(route),
        )
    }
    // The windows shrink across the vector crossover, so naming the selected engine would hide the tail.
    val bodies = (n - 1 downTo 1).map { panels.implementationFor(PanelWork.RankUpdate, it, it) }.distinct()
    val groups = (n - 1 downTo 1).map { panels.executionGroup(PanelWork.RankUpdate, it, it) }.distinct()
    val kind = if (bodies.size == 1) RouteKind.Direct else RouteKind.Composed
    return CaseWork(
        kind.name.lowercase(), "arithmetic", run, result = fixture.matrix.values,
        kernel = "${bodies.joinToString("+")}/rank-update-shrinking@${groups.joinToString("+")}",
    )
}
