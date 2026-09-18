package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.vendor.RouteKind
import com.eignex.koblas.vendor.BlasOperation

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
 * Level 2 or 3 work, named by the route the binding reports for this call.
 *
 * Null on a host with no library, and null for a call the binding would not serve directly, so a composed or
 * no-work route never reaches a row as if it were the vendor's arithmetic.
 */
private fun level23(
    engine: KoblasEngine,
    operation: BlasOperation,
    matrices: List<DenseMatrix>,
    timing: String,
    run: () -> Double,
): CaseWork? {
    val blas = engine.vendor ?: return null
    val route = blas.routeOf(operation, matrices, emptyList())
    if (route.kind != RouteKind.Direct) return null
    return CaseWork(route.kind.name.lowercase(), timing, run, kernel = vendorKernel(route))
}

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
        "gemv" -> {
            val m = d[0]; val n = d[1]; val trans = case.flag("transA")
            val a = Fixtures.matrix(if (trans) n else m, if (trans) m else n, 1)
            val x = Fixtures.vector(n, 2); val y0 = Fixtures.vector(m, 3); val y = y0.copyOf()
            level23(engine, BlasOperation.Gemv, listOf(a), "reset-and-arithmetic") {
                y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, trans); y[0]
            }
        }
        "symv" -> {
            val n = d[0]; val lower = case.option("uplo", "L") == "L"; val a = Fixtures.matrix(n, n, 1)
            val x = Fixtures.vector(n, 2); val y0 = Fixtures.vector(n, 3); val y = y0.copyOf()
            level23(engine, BlasOperation.Symv, listOf(a), "reset-and-arithmetic") {
                y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower); y[0]
            }
        }
        "ger" -> matrixUpdate(engine, BlasOperation.Ger, d[0], d[1]) { matrix, x, y ->
            engine.ger(alpha, x, y, matrix)
        }
        "syr" -> symmetricUpdate(engine, BlasOperation.Syr, d[0]) { matrix, x, _ ->
            engine.syr(alpha, DenseVector.wrap(x), matrix, case.option("uplo", "L") == "L")
        }
        "syr2" -> symmetricUpdate(engine, BlasOperation.Syr2, d[0]) { matrix, x, y ->
            engine.syr2(alpha, DenseVector.wrap(x), DenseVector.wrap(y), matrix, case.option("uplo", "L") == "L")
        }
        "trsv", "trmv" -> {
            val n = d[0]; val lower = case.option("uplo", "L") == "L"; val trans = case.flag("transA"); val unit = case.option("diag", "N") == "U"
            val a = Fixtures.triangular(n, 1, lower); val x0 = Fixtures.vector(n, 2); val x = x0.copyOf()
            val solve = case.operation == "trsv"
            val operation = if (solve) BlasOperation.Trsv else BlasOperation.Trmv
            level23(engine, operation, listOf(a), "reset-and-arithmetic") {
                x0.copyInto(x)
                if (solve) engine.trsv(a, x, lower, trans, unit) else engine.trmv(a, x, lower, trans, unit)
                x[0]
            }
        }
        "gemm" -> gemmWork(case, engine)
        "symm" -> {
            val m = d[0]; val n = d[1]; val right = case.option("side", "L") == "R"; val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (right) n else m, if (right) n else m, 1); val b = Fixtures.matrix(m, n, 2)
            val c0 = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3)
            level23(engine, BlasOperation.Symm, listOf(a, b, c), "reset-and-arithmetic") {
                c0.values.copyInto(c.values); engine.symm(alpha, a, b, beta, c, lower, right); c.values[0]
            }
        }
        "gemmt" -> {
            val n = d[0]; val k = d[1]; val ta = case.flag("transA"); val tb = case.flag("transB"); val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (ta) k else n, if (ta) n else k, 1); val b = Fixtures.matrix(if (tb) n else k, if (tb) k else n, 2)
            val c0 = Fixtures.matrix(n, n, 3); val c = Fixtures.matrix(n, n, 3)
            level23(engine, BlasOperation.Gemmt, listOf(a, b, c), "reset-and-arithmetic") {
                c0.values.copyInto(c.values); engine.gemmt(alpha, a, ta, b, tb, beta, c, lower); c.values[0]
            }
        }
        "syrk", "syr2k" -> {
            val n = d[0]; val k = d[1]; val trans = case.flag("transA"); val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (trans) k else n, if (trans) n else k, 1); val b = Fixtures.matrix(a.rows, a.cols, 2)
            val c0 = Fixtures.matrix(n, n, 3); val c = Fixtures.matrix(n, n, 3)
            val single = case.operation == "syrk"
            val operation = if (single) BlasOperation.Syrk else BlasOperation.Syr2k
            val operands = if (single) listOf(a, c) else listOf(a, b, c)
            level23(engine, operation, operands, "reset-and-arithmetic") {
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

private fun matrixUpdate(
    engine: KoblasEngine,
    operation: BlasOperation,
    rows: Int,
    cols: Int,
    run: (DenseMatrix, DoubleArray, DoubleArray) -> Unit,
): CaseWork? {
    val original = Fixtures.matrix(rows, cols, 1); val target = Fixtures.matrix(rows, cols, 1)
    val x = Fixtures.vector(rows, 2); val y = Fixtures.vector(cols, 3)
    return level23(engine, operation, listOf(target), "reset-and-arithmetic") {
        original.values.copyInto(target.values); run(target, x, y); target.values[0]
    }
}

private fun symmetricUpdate(
    engine: KoblasEngine,
    operation: BlasOperation,
    size: Int,
    run: (DenseMatrix, DoubleArray, DoubleArray) -> Unit,
): CaseWork? {
    val original = Fixtures.matrix(size, size, 1); val target = Fixtures.matrix(size, size, 1)
    val x = Fixtures.vector(size, 2); val y = Fixtures.vector(size, 3)
    return level23(engine, operation, listOf(target), "reset-and-arithmetic") {
        original.values.copyInto(target.values); run(target, x, y); target.values[0]
    }
}

private fun gemmWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val (m, n, k) = case.dimensions; val ta = case.flag("transA"); val tb = case.flag("transB")
    val a = Fixtures.matrix(if (ta) k else m, if (ta) m else k, 1)
    val b = Fixtures.matrix(if (tb) n else k, if (tb) k else n, 2)
    val original = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3)
    return level23(engine, BlasOperation.Gemm, listOf(a, b, c), "reset-and-arithmetic") {
        original.values.copyInto(c.values); engine.gemm(0.875, a, ta, b, tb, -0.25, c); c.values[0]
    }
}

private fun triangularMatrixWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    val (m, n) = case.dimensions; val right = case.option("side", "L") == "R"; val lower = case.option("uplo", "L") == "L"
    val trans = case.flag("transA"); val unit = case.option("diag", "N") == "U"; val order = if (right) n else m
    val triangle = Fixtures.triangular(order, 1, lower); val original = Fixtures.matrix(m, n, 2); val b = Fixtures.matrix(m, n, 2)
    val solve = case.operation == "trsm"
    val operation = if (solve) BlasOperation.Trsm else BlasOperation.Trmm
    return level23(engine, operation, listOf(triangle, b), "reset-and-arithmetic") {
        original.values.copyInto(b.values)
        if (solve) engine.trsm(triangle, b, lower, trans, unit, right, 0.875)
        else engine.trmm(triangle, b, lower, trans, unit, right, 0.875)
        b.values[0]
    }
}

