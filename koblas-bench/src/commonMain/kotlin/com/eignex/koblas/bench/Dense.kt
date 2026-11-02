package com.eignex.koblas.bench

import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Matrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

internal class CaseWork(
    val comparisonKind: String,
    val timingMode: String,
    val run: () -> Double,
    val close: () -> Unit = {},
    val result: DoubleArray? = null,
    /**
     * What the row was timed on. Declined cases have no call to describe; [measurement] requires one for every
     * timed row.
     */
    val kernel: String? = null,
)

/** The comparison an engine row makes: the engine its arm pinned, not a body a given width reached. */
internal const val ENGINE_COMPARISON = "engine"

/** The attribution an engine row carries, which names the selection rather than a reached body. */
internal fun engineKernel(engine: KoblasEngine, operation: String): String = "${engine.name}/$operation"

/** A timed row on [engine], attributed to the engine that was selected. */
internal fun engineWork(
    engine: KoblasEngine,
    operation: String,
    timing: String,
    run: () -> Double,
    result: DoubleArray? = null,
): CaseWork = CaseWork(ENGINE_COMPARISON, timing, run, result = result, kernel = engineKernel(engine, operation))

/** The reason a case with an empty extent is not timed. */
internal fun declined(): ArmChoice =
    ArmChoice(null, "this case's own contract stops before the arithmetic, so there is nothing to time")

/** Level 1 work, declined for an empty run. */
@Suppress("LongParameterList") // the operation, its width, and the timed region
private fun level1(
    engine: KoblasEngine,
    operation: String,
    length: Int,
    timing: String,
    result: DoubleArray? = null,
    run: () -> Double,
): CaseWork? = if (length == 0) null else engineWork(engine, operation, timing, run, result)

/**
 * Built-in Level 2 or 3 work, declined when one of the case's extents is empty.
 *
 * [verify] runs before anything is timed and compares the whole destination buffer against [DenseReference].
 */
private fun level23(
    engine: KoblasEngine,
    case: BenchCase,
    timing: String,
    verify: () -> Unit,
    run: () -> Double,
): CaseWork? {
    if (case.dimensions.any { it == 0 }) return null
    verify()
    return engineWork(engine, case.operation, timing, run)
}

@Suppress("LongMethod", "CyclomaticComplexMethod") // one branch per benchmarked operation
internal fun denseWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val d = case.dimensions
    val vectors = engine
    val alpha = 0.875
    val beta = -0.25
    return when (case.operation) {
        "dot" -> {
            val x = Fixtures.vector(d[0], 1); val y = Fixtures.vector(d[0], 2)
            level1(engine, case.operation, d[0], "arithmetic") { vectors.dot(x, 0, y, 0, d[0]) }
        }
        "axpy" -> {
            val x = Fixtures.vector(d[0], 1); val initial = Fixtures.vector(d[0], 2); val y = initial.copyOf()
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // Alternating the sign returns y close to where it started, so no reset joins the timed work.
                var signed = alpha
                level1(engine, case.operation, d[0], "arithmetic", result = y) {
                    signed = -signed; vectors.axpy(y, 0, signed, x, 0, d[0]); y[0]
                }
            } else {
                level1(engine, case.operation, d[0], "reset-and-arithmetic") {
                    initial.copyInto(y); vectors.axpy(y, 0, alpha, x, 0, d[0]); y[0]
                }
            }
        }
        "axpy-unit" -> {
            val length = d[0]
            val fixture = UnitAxpy(length)
            val x = fixture.x
            val y = fixture.y
            val run = {
                vectors.axpy(y, 0, 1.0, x, 0, length)
                y[0]
            }
            fixture.verify(case.id, run)
            level1(engine, case.operation, length, "arithmetic", result = y, run = run)
        }
        "scal" -> {
            val initial = Fixtures.vector(d[0], 1); val x = initial.copyOf()
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // Negation preserves normal magnitudes over arbitrarily many timed invocations.
                level1(engine, case.operation, d[0], "arithmetic", result = x) {
                    vectors.scale(x, 0, -1.0, d[0]); x[0] + x.last()
                }
            } else {
                level1(engine, case.operation, d[0], "reset-and-arithmetic", result = x) {
                    initial.copyInto(x); vectors.scale(x, 0, alpha, d[0]); x[0] + x.last()
                }
            }
        }
        "nrm2" -> vectorReduction(engine, case.operation, d[0]) { x -> vectors.nrm2(x, 0, x.size) }
        "asum" -> vectorReduction(engine, case.operation, d[0]) { x -> vectors.asum(x, 0, x.size) }
        "sum" -> vectorReduction(engine, case.operation, d[0]) { x -> vectors.sum(x, 0, x.size) }
        "iamax" -> vectorReduction(engine, case.operation, d[0]) { x -> vectors.iamax(x, 0, x.size).toDouble() }
        "swap" -> {
            val x0 = Fixtures.vector(d[0], 1); val y0 = Fixtures.vector(
                d[0],
                2,
            ); val x = x0.copyOf(); val y = y0.copyOf()
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // A swap is its own inverse, so the operands only ever trade places.
                level1(engine, case.operation, d[0], "arithmetic", result = y) {
                    vectors.swap(x, 0, y, 0, d[0]); x[0] + y[0]
                }
            } else {
                level1(engine, case.operation, d[0], "reset-and-arithmetic") {
                    x0.copyInto(x); y0.copyInto(y); vectors.swap(x, 0, y, 0, d[0]); x[0] + y[0]
                }
            }
        }
        "rot" -> {
            val x0 = Fixtures.vector(d[0], 1); val y0 = Fixtures.vector(
                d[0],
                2,
            ); val x = x0.copyOf(); val y = y0.copyOf()
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // A rotation preserves each pair's norm, so repeating it keeps the magnitudes normal.
                level1(engine, case.operation, d[0], "arithmetic", result = y) {
                    vectors.rot(x, 0, y, 0, d[0], 0.8, 0.6); x[0] + y[0]
                }
            } else {
                level1(engine, case.operation, d[0], "reset-and-arithmetic") {
                    x0.copyInto(x); y0.copyInto(y); vectors.rot(x, 0, y, 0, d[0], 0.8, 0.6); x[0] + y[0]
                }
            }
        }
        "gemv" -> {
            val m = d[0]; val n = d[1]; val trans = case.flag("transA")
            val a = Fixtures.matrix(if (trans) n else m, if (trans) m else n, 1)
            val x = Fixtures.vector(n, 2); val y0 = Fixtures.vector(m, 3); val y = y0.copyOf()
            val resets = resets(case)
            level23(
                engine, case, timing(case),
                verify = {
                    val expected = DenseReference.gemv(alpha, a, trans, x, beta, y0)
                    y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, trans)
                    DenseReference.check(expected, y, "${case.id} gemv")
                },
            ) { if (resets) y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, trans); y[0] }
        }
        "symv" -> {
            val n = d[0]; val lower = case.option("uplo", "L") == "L"; val a = Fixtures.matrix(n, n, 1)
            val x = Fixtures.vector(n, 2); val y0 = Fixtures.vector(n, 3); val y = y0.copyOf()
            val resets = resets(case)
            level23(
                engine, case, timing(case),
                verify = {
                    val expected = DenseReference.symv(alpha, a, lower, x, beta, y0)
                    y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower)
                    DenseReference.check(expected, y, "${case.id} symv")
                },
            ) { if (resets) y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower); y[0] }
        }
        "ger", "syr", "syr2" -> rankUpdateWork(case, engine)
        "ger-shrinking", "ger-window-shrinking" -> shrinkingRankUpdateWork(case, engine)
        "trsv", "trmv" -> triangularVectorWork(case, engine)
        "gemm" -> gemmWork(case, engine)
        "symm" -> {
            val m = d[0]; val n = d[1]; val right = case.option("side", "L") == "R"; val lower = case.option(
                "uplo",
                "L",
            ) == "L"
            val a = Fixtures.matrix(if (right) n else m, if (right) n else m, 1); val b = Fixtures.matrix(m, n, 2)
            val c0 = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3)
            level23(
                engine, case, "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.symm(alpha, a, lower, right, b, beta, c0)
                    c0.values.copyInto(c.values); engine.symm(alpha, a, b, beta, c, lower, right)
                    DenseReference.check(expected, c.values, "${case.id} symm")
                },
            ) { c0.values.copyInto(c.values); engine.symm(alpha, a, b, beta, c, lower, right); c.values[0] }
        }
        "gemmt" -> {
            val n = d[0]; val k = d[1]; val ta = case.flag("transA"); val tb = case.flag("transB"); val lower = case.option(
                "uplo",
                "L",
            ) == "L"
            val a = Fixtures.matrix(if (ta) k else n, if (ta) n else k, 1); val b = Fixtures.matrix(if (tb) n else k, if (tb) k else n, 2)
            val c0 = Fixtures.matrix(n, n, 3); val c = Fixtures.matrix(n, n, 3)
            level23(
                engine, case, "reset-and-arithmetic",
                verify = {
                    val expected = DenseReference.gemmt(alpha, a, ta, b, tb, beta, c0, lower)
                    c0.values.copyInto(c.values); engine.gemmt(alpha, a, ta, b, tb, beta, c, lower)
                    DenseReference.check(expected, c.values, "${case.id} gemmt")
                },
            ) { c0.values.copyInto(c.values); engine.gemmt(alpha, a, ta, b, tb, beta, c, lower); c.values[0] }
        }
        "syrk", "syr2k" -> {
            val n = d[0]; val k = d[1]; val trans = case.flag("transA"); val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (trans) k else n, if (trans) n else k, 1); val b = Fixtures.matrix(
                a.rows,
                a.cols,
                2,
            )
            val c0 = Fixtures.matrix(n, n, 3); val c = Fixtures.matrix(n, n, 3)
            val single = case.operation == "syrk"
            level23(
                engine, case, "reset-and-arithmetic",
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
    operation: String,
    size: Int,
    run: (DoubleArray) -> Double,
): CaseWork? {
    val x = Fixtures.vector(size, 1)
    return level1(engine, operation, size, "arithmetic") { run(x) }
}

/**
 * The dense work for [case], or a reason this arm may not measure it.
 *
 * Raw cases run on the engine the arm pinned. Generic matrix and vector entry
 * points select their engine themselves, so those cases are published as default-policy rows.
 */
internal fun denseArm(case: BenchCase, engine: KoblasEngine): ArmChoice? = when (case.operation) {
    "gemm-generic" -> genericProductArm(case, engine)
    "copy-overlap", "axpy-views", "swap-views" -> vectorViewsArm(case, engine)
    else -> denseWork(case, engine)?.let { ArmChoice(it, null) }
}

private fun genericProductArm(case: BenchCase, engine: KoblasEngine): ArmChoice {
    val (m, n, k) = case.dimensions
    val alpha = 0.875
    val beta = -0.25
    val a = Fixtures.matrix(m, k, 1)
    val b = Fixtures.matrix(k, n, 2)
    val c0 = Fixtures.matrix(m, n, 3)
    val c = Fixtures.matrix(m, n, 3)
    if (case.dimensions.any { it == 0 }) return declined()
    val expected = DenseReference.gemm(alpha, a, false, b, false, beta, c0)
    c0.values.copyInto(c.values)
    engine.gemm(alpha, a as Matrix, false, b as Matrix, false, beta, c)
    DenseReference.check(expected, c.values, "${case.id} generic product")
    return ArmChoice(
        engineWork(engine, case.operation, "reset-and-arithmetic", {
            c0.values.copyInto(c.values)
            engine.gemm(alpha, a as Matrix, false, b as Matrix, false, beta, c)
            c.values[0]
        }),
        null,
    )
}

private fun rankUpdateWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val alpha = 0.875
    val lower = case.option("uplo", "L") == "L"
    val rows = case.dimension(0)
    val cols = if (case.operation == "ger") case.dimension(1) else rows
    val original = Fixtures.matrix(rows, cols, 1); val target = Fixtures.matrix(rows, cols, 1)
    val x = Fixtures.vector(rows, 2); val y = Fixtures.vector(cols, 3)
    val operation = case.operation
    fun apply(scale: Double) = when (operation) {
        "ger" -> engine.ger(scale, x, y, target)
        "syr" -> engine.syr(scale, DenseVector.wrap(x), target, lower)
        else -> engine.syr2(scale, DenseVector.wrap(x), DenseVector.wrap(y), target, lower)
    }
    val resets = resets(case)
    // Without the reset, alternating the sign returns the matrix close to where it started.
    var signed = -alpha
    return level23(
        engine, case, timing(case),
        verify = {
            val expected = when (operation) {
                "ger" -> DenseReference.ger(alpha, x, y, original)
                "syr" -> DenseReference.syr(alpha, x, original, lower)
                else -> DenseReference.syr2(alpha, x, y, original, lower)
            }
            original.values.copyInto(target.values); apply(alpha)
            DenseReference.check(expected, target.values, "${case.id} ${case.operation}")
            original.values.copyInto(target.values)
        },
    ) {
        if (resets) original.values.copyInto(target.values) else signed = -signed
        apply(if (resets) alpha else signed); target.values[0]
    }
}

private fun triangularVectorWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val n = case.dimension(0)
    val lower = case.option("uplo", "L") == "L"
    val trans = case.flag("transA")
    val unit = case.option("diag", "N") == "U"
    val a = Fixtures.triangular(n, 1, lower); val x0 = Fixtures.vector(n, 2); val x = x0.copyOf()
    val solve = case.operation == "trsv"
    fun apply() = if (solve) engine.trsv(a, x, lower, trans, unit) else engine.trmv(a, x, lower, trans, unit)
    return level23(
                engine, case, "reset-and-arithmetic",
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
    if (case.options["layout"] == "borrowed") return borrowedProductWork(case, engine)
    val (m, n, k) = case.dimensions; val ta = case.flag("transA"); val tb = case.flag("transB")
    val alpha = 0.875; val beta = -0.25
    val a = Fixtures.matrix(if (ta) k else m, if (ta) m else k, 1)
    val b = Fixtures.matrix(if (tb) n else k, if (tb) k else n, 2)
    val original = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3)
    val workspace = if (case.options["workspace"] == "retained") Workspace() else null
    return level23(
        engine, case,
        "reset-and-arithmetic",
        verify = {
            val expected = DenseReference.gemm(alpha, a, ta, b, tb, beta, original)
            original.values.copyInto(c.values); engine.gemm(alpha, a, ta, b, tb, beta, c, workspace)
            DenseReference.check(expected, c.values, "${case.id} gemm")
        },
    ) { original.values.copyInto(c.values); engine.gemm(alpha, a, ta, b, tb, beta, c, workspace); c.values[0] }
}

private fun triangularMatrixWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val (m, n) = case.dimensions; val right = case.option("side", "L") == "R"; val lower = case.option(
        "uplo",
        "L",
    ) == "L"
    val trans = case.flag("transA"); val unit = case.option("diag", "N") == "U"; val order = if (right) n else m
    val alpha = 0.875
    val triangle = Fixtures.triangular(order, 1, lower); val original = Fixtures.matrix(m, n, 2); val b = Fixtures.matrix(
        m,
        n,
        2,
    )
    val solve = case.operation == "trsm"
    fun apply() = if (solve) {
        engine.trsm(triangle, b, lower, trans, unit, right, alpha)
    } else {
        engine.trmm(triangle, b, lower, trans, unit, right, alpha)
    }
    return level23(
        engine, case, "reset-and-arithmetic",
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

/**
 * The timing a case asked for. Its arithmetic form leaves the reset out of the timed work, which an operation
 * offers where repeating it keeps the destination's magnitudes normal, as a nonzero beta below one does.
 */
internal fun timing(case: BenchCase): String = case.option("timing", "reset-and-arithmetic")

/** Whether each timed call first restores its destination, read once so the timed work does not parse the case. */
internal fun resets(case: BenchCase): Boolean = timing(case) != "arithmetic"
