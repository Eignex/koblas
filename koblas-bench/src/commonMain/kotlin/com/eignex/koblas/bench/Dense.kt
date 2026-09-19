package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.DenseMatrixRoute
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.PanelWork
import com.eignex.koblas.vendor.RouteKind

internal class CaseWork(
    val comparisonKind: String,
    val timingMode: String,
    val run: () -> Double,
    val close: () -> Unit = {},
    val result: DoubleArray? = null,
    /**
     * What ran, as the route of the call this work makes reported it.
     *
     * Attribution taken from here comes from the same decision the timed call acts on. That is the difference
     * from a kernel name rebuilt afterwards out of the mode and case strings, which is a second answer to the
     * same question and can disagree with the first. It stays nullable because a declined case has no call to
     * describe; a timed one with nothing here is refused by `measurement` rather than reported.
     */
    val kernel: String? = null,
)

/**
 * Level 1 work, named by the kernel selection that will serve it.
 *
 * Null where the kernel's own values settle which implementation finishes the call, as the euclidean norm's
 * rescaling retry does. No width establishes that, so the case is not an exact measurement of either kernel
 * and is declined rather than attributed to the one that happened to start it.
 */
@Suppress("LongParameterList") // the operation, its width, and the timed region
private fun level1(
    engine: KoblasEngine,
    operation: DenseOperation,
    length: Int,
    timing: String,
    result: DoubleArray? = null,
    run: () -> Double,
): CaseWork? {
    val implementation = engine.explain(operation, length) ?: return null
    return CaseWork("direct", timing, run, result = result, kernel = "$implementation/${operation.name.lowercase()}")
}

/**
 * Built-in Level 2 or 3 work, named by the route the call itself resolves.
 *
 * The scheduling is this library's own portable code on every engine, and the route says so; where a window
 * of work reaches a panel or a Level 1 kernel, the route names the one that window reaches at its own length.
 * An arm therefore never publishes a scalar traversal under a SIMD label because the engine selected
 * vectorised kernels, and a call whose windows do not all reach the same body is published as the composition
 * it is rather than under either name.
 *
 * [verify] runs before anything is timed and compares the whole destination buffer against [DenseReference].
 */
@Suppress("LongParameterList") // the operation, the facts its route needs, and both regions
private fun level23(
    engine: KoblasEngine,
    operation: DenseMatrixOperation,
    call: DenseCall,
    timing: String,
    verify: () -> Unit,
    run: () -> Double,
): CaseWork? {
    val route = engine.denseRouteOf(operation, call)
    if (route.kind == RouteKind.NoWork) return null
    verify()
    return CaseWork(route.kind.name.lowercase(), timing, run, kernel = denseMatrixKernel(route))
}

/** The attribution a dense matrix row carries, taken from the route the call resolved. */
internal fun denseMatrixKernel(route: DenseMatrixRoute): String =
    "${route.implementation}/${route.entryPoint}" + if (route.executionGroup > 0) "@${route.executionGroup}" else ""

/**
 * Raw panel work, named by the implementation that panel's own extents reach and the group it was given.
 *
 * A raw case hands the whole logical panel over in one call, so the grouping inside it is the backend's own
 * and the row records which one that was. The extents are the case's and do not depend on the backend, which
 * is what keeps a tail at three columns comparable across arms that group by two and by four.
 */
private fun panel(
    engine: KoblasEngine,
    work: PanelWork,
    rows: Int,
    columns: Int,
    timing: String,
    verify: () -> Unit,
    run: () -> Double,
): CaseWork {
    val panels = engine.panelKernels
    val group = panels.executionGroup(work, rows, columns)
    val kernel = "${panels.implementationFor(work, rows, columns)}/${panelEntryPoint(work)}@$group"
    verify()
    return CaseWork("direct", timing, run, kernel = kernel)
}

private fun panelEntryPoint(work: PanelWork): String = when (work) {
    PanelWork.MultiDot -> "multi-dot"
    PanelWork.ColumnUpdate -> "column-update"
    PanelWork.CoupledDotUpdate -> "coupled-dot-update"
    PanelWork.RankUpdate -> "rank-update"
    PanelWork.SparseRightHandSides -> "sparse-rhs"
}

@Suppress("LongMethod", "CyclomaticComplexMethod") // one branch per benchmarked operation
internal fun denseWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val d = case.dimensions
    val vectors = engine.vectorKernels
    val alpha = 0.875
    val beta = -0.25
    return when (case.operation) {
        "dot" -> {
            val x = Fixtures.vector(d[0], 1); val y = Fixtures.vector(d[0], 2)
            level1(engine, DenseOperation.Dot, d[0], "arithmetic") { vectors.dot(x, 0, y, 0, d[0]) }
        }
        "axpy" -> {
            val x = Fixtures.vector(d[0], 1); val initial = Fixtures.vector(d[0], 2); val y = initial.copyOf()
            level1(engine, DenseOperation.Axpy, d[0], "reset-and-arithmetic") {
                initial.copyInto(y); vectors.axpy(y, 0, alpha, x, 0, d[0]); y[0]
            }
        }
        "scal" -> {
            val initial = Fixtures.vector(d[0], 1); val x = initial.copyOf()
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // Negation preserves normal magnitudes over arbitrarily many timed invocations.
                level1(engine, DenseOperation.Scale, d[0], "arithmetic", result = x) {
                    vectors.scale(x, 0, -1.0, d[0]); x[0] + x.last()
                }
            } else {
                level1(engine, DenseOperation.Scale, d[0], "reset-and-arithmetic", result = x) {
                    initial.copyInto(x); vectors.scale(x, 0, alpha, d[0]); x[0] + x.last()
                }
            }
        }
        "nrm2" -> vectorReduction(engine, DenseOperation.Nrm2, d[0]) { x -> vectors.nrm2(x, 0, x.size) }
        "asum" -> vectorReduction(engine, DenseOperation.Asum, d[0]) { x -> vectors.asum(x, 0, x.size) }
        "sum" -> vectorReduction(engine, DenseOperation.Sum, d[0]) { x -> vectors.sum(x, 0, x.size) }
        "iamax" -> vectorReduction(engine, DenseOperation.Iamax, d[0]) { x -> vectors.iamax(x, 0, x.size).toDouble() }
        "swap" -> {
            val x0 = Fixtures.vector(d[0], 1); val y0 = Fixtures.vector(d[0], 2); val x = x0.copyOf(); val y = y0.copyOf()
            level1(engine, DenseOperation.Swap, d[0], "reset-and-arithmetic") {
                x0.copyInto(x); y0.copyInto(y); vectors.swap(x, 0, y, 0, d[0]); x[0] + y[0]
            }
        }
        "rot" -> {
            val x0 = Fixtures.vector(d[0], 1); val y0 = Fixtures.vector(d[0], 2); val x = x0.copyOf(); val y = y0.copyOf()
            level1(engine, DenseOperation.Rot, d[0], "reset-and-arithmetic") {
                x0.copyInto(x); y0.copyInto(y); vectors.rot(x, 0, y, 0, d[0], 0.8, 0.6); x[0] + y[0]
            }
        }
        "panel-multidot", "panel-columnupdate", "panel-coupled", "panel-rankupdate" -> panelWork(case, engine)
        "gemv" -> {
            val m = d[0]; val n = d[1]; val trans = case.flag("transA")
            val a = Fixtures.matrix(if (trans) n else m, if (trans) m else n, 1)
            val x = Fixtures.vector(n, 2); val y0 = Fixtures.vector(m, 3); val y = y0.copyOf()
            val operation = if (trans) DenseMatrixOperation.GemvTransposed else DenseMatrixOperation.Gemv
            level23(
                engine, operation, DenseCall(a.rows, a.cols, alpha, beta), "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.gemv(alpha, a, trans, x, beta, y0)
                    y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, trans)
                    DenseReference.check(expected, y, "${case.id} gemv")
                },
            ) { y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, trans); y[0] }
        }
        "symv" -> {
            val n = d[0]; val lower = case.option("uplo", "L") == "L"; val a = Fixtures.matrix(n, n, 1)
            val x = Fixtures.vector(n, 2); val y0 = Fixtures.vector(n, 3); val y = y0.copyOf()
            level23(
                engine, DenseMatrixOperation.Symv, DenseCall(n, n, alpha, beta, lower = lower), "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.symv(alpha, a, lower, x, beta, y0)
                    y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower)
                    DenseReference.check(expected, y, "${case.id} symv")
                },
            ) { y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower); y[0] }
        }
        "ger", "syr", "syr2" -> rankUpdateWork(case, engine)
        "trsv", "trmv" -> triangularVectorWork(case, engine)
        "gemm" -> gemmWork(case, engine)
        "symm" -> {
            val m = d[0]; val n = d[1]; val right = case.option("side", "L") == "R"; val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (right) n else m, if (right) n else m, 1); val b = Fixtures.matrix(m, n, 2)
            val c0 = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3)
            level23(
                engine, DenseMatrixOperation.Symm, DenseCall(m, n, alpha, beta, depth = a.rows, lower = lower),
                "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.symm(alpha, a, lower, right, b, beta, c0)
                    c0.values.copyInto(c.values); engine.symm(alpha, a, b, beta, c, lower, right)
                    DenseReference.check(expected, c.values, "${case.id} symm")
                },
            ) { c0.values.copyInto(c.values); engine.symm(alpha, a, b, beta, c, lower, right); c.values[0] }
        }
        "gemmt" -> {
            val n = d[0]; val k = d[1]; val ta = case.flag("transA"); val tb = case.flag("transB"); val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (ta) k else n, if (ta) n else k, 1); val b = Fixtures.matrix(if (tb) n else k, if (tb) k else n, 2)
            val c0 = Fixtures.matrix(n, n, 3); val c = Fixtures.matrix(n, n, 3)
            level23(
                engine, DenseMatrixOperation.Gemmt, DenseCall(n, n, alpha, beta, depth = k, lower = lower), "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.gemmt(alpha, a, ta, b, tb, beta, c0, lower)
                    c0.values.copyInto(c.values); engine.gemmt(alpha, a, ta, b, tb, beta, c, lower)
                    DenseReference.check(expected, c.values, "${case.id} gemmt")
                },
            ) { c0.values.copyInto(c.values); engine.gemmt(alpha, a, ta, b, tb, beta, c, lower); c.values[0] }
        }
        "syrk", "syr2k" -> {
            val n = d[0]; val k = d[1]; val trans = case.flag("transA"); val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (trans) k else n, if (trans) n else k, 1); val b = Fixtures.matrix(a.rows, a.cols, 2)
            val c0 = Fixtures.matrix(n, n, 3); val c = Fixtures.matrix(n, n, 3)
            val single = case.operation == "syrk"
            val operation = if (single) DenseMatrixOperation.Syrk else DenseMatrixOperation.Syr2k
            level23(
                engine, operation, DenseCall(n, n, alpha, beta, depth = k, lower = lower), "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.syrk(alpha, a, b, trans, beta, c0, lower, doubled = !single)
                    c0.values.copyInto(c.values)
                    if (single) engine.syrk(alpha, a, trans, beta, c, lower)
                    else engine.syr2k(alpha, a, b, trans, beta, c, lower)
                    DenseReference.check(expected, c.values, "${case.id} ${case.operation}")
                },
            ) {
                c0.values.copyInto(c.values)
                if (single) engine.syrk(alpha, a, trans, beta, c, lower)
                else engine.syr2k(alpha, a, b, trans, beta, c, lower)
                c.values[0]
            }
        }
        "trsm", "trmm" -> triangularMatrixWork(case, engine)
        else -> null
    }
}

private fun vectorReduction(
    engine: KoblasEngine,
    operation: DenseOperation,
    size: Int,
    run: (DoubleArray) -> Double,
): CaseWork? {
    val x = Fixtures.vector(size, 1)
    return level1(engine, operation, size, "arithmetic") { run(x) }
}

/** A raw panel over extents the case states, with its own operands and its own textbook check. */
private fun panelWork(case: BenchCase, engine: KoblasEngine): CaseWork {
    val rows = case.dimension(0)
    val columns = case.dimension(1)
    val alpha = 0.875
    val panels = engine.panelKernels
    val a = Fixtures.vector(rows * columns, 1)
    val x = Fixtures.vector(maxOf(rows, columns), 2)
    val coefficients = Fixtures.vector(maxOf(rows, columns), 4)
    return when (case.operation) {
        "panel-multidot" -> {
            val y = DoubleArray(columns)
            panel(
                engine, PanelWork.MultiDot, rows, columns, "arithmetic",
                verify = {
                    // A zero beta overwrites, so the destination needs no reset between repetitions.
                    val expected = DenseReference.multiDot(alpha, a, rows, columns, x, 0.0, y)
                    panels.multiDot(alpha, a, 0, rows, x, 0, 1, rows, columns, 0.0, y, 0, 1)
                    DenseReference.check(expected, y, case.id)
                },
            ) { panels.multiDot(alpha, a, 0, rows, x, 0, 1, rows, columns, 0.0, y, 0, 1); y[0] }
        }
        "panel-columnupdate" -> {
            val y0 = Fixtures.vector(rows, 3); val y = y0.copyOf()
            panel(
                engine, PanelWork.ColumnUpdate, rows, columns, "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.columnUpdate(alpha, a, rows, columns, x, y0)
                    y0.copyInto(y); panels.columnUpdate(alpha, a, 0, rows, x, 0, 1, rows, columns, y, 0, 1)
                    DenseReference.check(expected, y, case.id)
                },
            ) { y0.copyInto(y); panels.columnUpdate(alpha, a, 0, rows, x, 0, 1, rows, columns, y, 0, 1); y[0] }
        }
        "panel-coupled" -> {
            val y0 = Fixtures.vector(rows, 3); val y = y0.copyOf()
            val sums0 = Fixtures.vector(columns, 5); val sums = sums0.copyOf()
            panel(
                engine, PanelWork.CoupledDotUpdate, rows, columns, "reset-and-arithmetic",
                verify = {
                    val expected =
                        DenseReference.coupledUpdateDot(alpha, a, rows, columns, x, coefficients, y0, sums0)
                    y0.copyInto(y); sums0.copyInto(sums)
                    panels.coupledUpdateDot(alpha, a, 0, rows, x, 0, rows, columns, y, 0, coefficients, 0, sums, 0)
                    DenseReference.check(expected, y + sums, case.id)
                },
            ) {
                y0.copyInto(y); sums0.copyInto(sums)
                panels.coupledUpdateDot(alpha, a, 0, rows, x, 0, rows, columns, y, 0, coefficients, 0, sums, 0)
                y[0] + sums[0]
            }
        }
        else -> {
            val target = a.copyOf()
            panel(
                engine, PanelWork.RankUpdate, rows, columns, "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.rankUpdate(alpha, a, rows, columns, x, coefficients)
                    a.copyInto(target)
                    panels.rankUpdate(alpha, target, 0, rows, x, 0, 1, rows, columns, coefficients, 0, 1)
                    DenseReference.check(expected, target, case.id)
                },
            ) {
                a.copyInto(target)
                panels.rankUpdate(alpha, target, 0, rows, x, 0, 1, rows, columns, coefficients, 0, 1)
                target[0]
            }
        }
    }
}

private fun rankUpdateWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val alpha = 0.875
    val lower = case.option("uplo", "L") == "L"
    val rows = case.dimension(0)
    val cols = if (case.operation == "ger") case.dimension(1) else rows
    val original = Fixtures.matrix(rows, cols, 1); val target = Fixtures.matrix(rows, cols, 1)
    val x = Fixtures.vector(rows, 2); val y = Fixtures.vector(cols, 3)
    val operation = when (case.operation) {
        "ger" -> DenseMatrixOperation.Ger
        "syr" -> DenseMatrixOperation.Syr
        else -> DenseMatrixOperation.Syr2
    }
    fun apply() = when (operation) {
        DenseMatrixOperation.Ger -> engine.ger(alpha, x, y, target)
        DenseMatrixOperation.Syr -> engine.syr(alpha, DenseVector.wrap(x), target, lower)
        else -> engine.syr2(alpha, DenseVector.wrap(x), DenseVector.wrap(y), target, lower)
    }
    return level23(
        engine, operation, DenseCall(rows, cols, alpha, lower = lower), "reset-and-arithmetic",
        verify = {
            val expected = when (operation) {
                DenseMatrixOperation.Ger -> DenseReference.ger(alpha, x, y, original)
                DenseMatrixOperation.Syr -> DenseReference.syr(alpha, x, original, lower)
                else -> DenseReference.syr2(alpha, x, y, original, lower)
            }
            original.values.copyInto(target.values); apply()
            DenseReference.check(expected, target.values, "${case.id} ${case.operation}")
        },
    ) { original.values.copyInto(target.values); apply(); target.values[0] }
}

private fun triangularVectorWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val n = case.dimension(0)
    val lower = case.option("uplo", "L") == "L"
    val trans = case.flag("transA")
    val unit = case.option("diag", "N") == "U"
    val a = Fixtures.triangular(n, 1, lower); val x0 = Fixtures.vector(n, 2); val x = x0.copyOf()
    val solve = case.operation == "trsv"
    val operation = when {
        solve && trans -> DenseMatrixOperation.TrsvTransposed
        solve -> DenseMatrixOperation.Trsv
        trans -> DenseMatrixOperation.TrmvTransposed
        else -> DenseMatrixOperation.Trmv
    }
    fun apply() = if (solve) engine.trsv(a, x, lower, trans, unit) else engine.trmv(a, x, lower, trans, unit)
    return level23(
        engine, operation, DenseCall(n, n, lower = lower), "reset-and-arithmetic",
        verify = {
            val expected = if (solve) {
                DenseReference.trsv(a, x0, lower, trans, unit)
            } else {
                DenseReference.trmv(a, x0, lower, trans, unit)
            }
            x0.copyInto(x); apply()
            DenseReference.check(expected, x, "${case.id} ${case.operation}")
        },
    ) { x0.copyInto(x); apply(); x[0] }
}

private fun gemmWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val (m, n, k) = case.dimensions; val ta = case.flag("transA"); val tb = case.flag("transB")
    val alpha = 0.875; val beta = -0.25
    val a = Fixtures.matrix(if (ta) k else m, if (ta) m else k, 1)
    val b = Fixtures.matrix(if (tb) n else k, if (tb) k else n, 2)
    val original = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3)
    return level23(
        engine, DenseMatrixOperation.Gemm, DenseCall(m, n, alpha, beta, depth = k), "reset-and-arithmetic",
        verify = {
            val expected = DenseReference.gemm(alpha, a, ta, b, tb, beta, original)
            original.values.copyInto(c.values); engine.gemm(alpha, a, ta, b, tb, beta, c)
            DenseReference.check(expected, c.values, "${case.id} gemm")
        },
    ) { original.values.copyInto(c.values); engine.gemm(alpha, a, ta, b, tb, beta, c); c.values[0] }
}

private fun triangularMatrixWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val (m, n) = case.dimensions; val right = case.option("side", "L") == "R"; val lower = case.option("uplo", "L") == "L"
    val trans = case.flag("transA"); val unit = case.option("diag", "N") == "U"; val order = if (right) n else m
    val alpha = 0.875
    val triangle = Fixtures.triangular(order, 1, lower); val original = Fixtures.matrix(m, n, 2); val b = Fixtures.matrix(m, n, 2)
    val solve = case.operation == "trsm"
    val operation = if (solve) DenseMatrixOperation.Trsm else DenseMatrixOperation.Trmm
    fun apply() = if (solve) {
        engine.trsm(triangle, b, lower, trans, unit, right, alpha)
    } else {
        engine.trmm(triangle, b, lower, trans, unit, right, alpha)
    }
    return level23(
        engine, operation, DenseCall(m, n, alpha, depth = order, lower = lower), "reset-and-arithmetic",
        verify = {
            val expected = if (solve) {
                DenseReference.trsm(triangle, original, lower, trans, unit, right, alpha)
            } else {
                DenseReference.trmm(triangle, original, lower, trans, unit, right, alpha)
            }
            original.values.copyInto(b.values); apply()
            DenseReference.check(expected, b.values, "${case.id} ${case.operation}")
        },
    ) { original.values.copyInto(b.values); apply(); b.values[0] }
}
