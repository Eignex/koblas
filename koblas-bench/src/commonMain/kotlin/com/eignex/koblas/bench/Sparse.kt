package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Matrix
import com.eignex.koblas.MatrixWorkspace
import com.eignex.koblas.PreparedSparseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.gemmInto
import com.eignex.koblas.sparse.SparseMatrixOperation
import com.eignex.koblas.sparse.SparseMatrixRoute
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.sparse.SparsePrimitives
import com.eignex.koblas.sparse.SparseRoute
import com.eignex.koblas.times
import kotlin.math.abs

/**
 * The sparse work for one case on [engine], or the reason this engine cannot measure it, or null when the case
 * is not a sparse one.
 *
 * The engine-dispatched arms are exact comparisons: the row is published under the engine that was selected, so
 * a call the engine hands to a different implementation is declined rather than timed. A Vector API selection
 * does that for a support below its crossover, for an operation it never vectorised, and on a host whose indexed
 * loads or stores it cannot use, and the route is what distinguishes those from a call it really ran.
 *
 * The generic primitives are the exception, because no engine selects them. They are one implementation, timed
 * once as an explicit composed case rather than repeated under each engine label.
 */
@Suppress("LongMethod", "CyclomaticComplexMethod") // one branch per benchmarked operation
internal fun sparseArm(case: BenchCase, engine: KoblasEngine): ArmChoice? {
    if (!case.operation.startsWith("sp")) return null
    val density = case.option("density", "0.01").toDouble()
    val d = case.dimensions
    val kernels = engine.sparseKernels

    // Resolved and checked once, outside the timed region, so no route work reaches a measured loop.
    fun arm(
        operation: SparseOperation,
        count: Int,
        timing: String,
        result: DoubleArray? = null,
        run: () -> Double,
    ): ArmChoice {
        val route = kernels.routeOf(operation, count)
        if (!route.exactlyMeasurable) return ArmChoice(null, requireNotNull(route.reason))
        val work = CaseWork(
            route.kind.name.lowercase(), timing, run, result = result, kernel = sparseKernel(route),
        )
        return ArmChoice(work, null)
    }

    return when (case.operation) {
        "spaccumulate" -> {
            // The generic primitives are one implementation for every engine, so this is timed once rather
            // than repeated under each label, where it would compare a selection against a copy of itself.
            if (kernels.name != "scalar") {
                return ArmChoice(null, "the generic primitives have one implementation, timed on the scalar arm")
            }
            val x = Fixtures.sparseVector(d[0], density, 1)
            val indices = x.copyIndices()
            val accumulator = DoubleArray(d[0])
            val marks = IntArray(d[0])
            val touched = IntArray(d[0])
            val outIndices = IntArray(d[0])
            val outValues = DoubleArray(d[0])
            // The gather clears what the scatter touched, so the workspace returns to its starting state and
            // the measured region needs no reset beside the arithmetic.
            val work = CaseWork("composed", "arithmetic", {
                val count = SparsePrimitives.scatterWorkspace(
                    0.875, indices, 0, x.values, 0, x.values.size,
                    accumulator, marks, EPOCH, touched, 0, 0,
                )
                val written = SparsePrimitives.gatherWorkspace(
                    touched, 0, count, accumulator, outIndices, 0, outValues, 0,
                    compactExactZeros = true, marks = marks,
                )
                written.toDouble()
            }, kernel = "primitives/scatterWorkspace plus gatherWorkspace")
            ArmChoice(work, null)
        }
        "spdot" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.vector(d[0], 2)
            arm(SparseOperation.DotDense, x.values.size, "arithmetic") { kernels.dot(x, y) }
        }
        "spdot-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.vector(d[0], 2)
            val indices = x.copyIndices()
            arm(SparseOperation.DotDense, x.values.size, "arithmetic") {
                kernels.dot(indices, 0, x.values, 0, x.values.size, y)
            }
        }
        "spdot-sparse" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.sparseVector(d[0], density, 2)
            arm(SparseOperation.DotSparse, x.values.size, "arithmetic") { kernels.dot(x, y) }
        }
        "spaxpy" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            arm(SparseOperation.Axpy, x.values.size, "reset-and-arithmetic") {
                y0.copyInto(y); kernels.axpy(y, 0.875, x); y[0]
            }
        }
        "spaxpy-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            arm(SparseOperation.Axpy, x.values.size, "reset-and-arithmetic") {
                y0.copyInto(y)
                kernels.axpy(y, 0.875, indices, 0, x.values, 0, x.values.size)
                y[0]
            }
        }
        "spnrm2", "spasum" -> {
            val x = Fixtures.sparseVector(d[0], density, 1)
            val operation = if (case.operation == "spnrm2") SparseOperation.Nrm2 else SparseOperation.Asum
            arm(operation, x.values.size, "arithmetic") {
                if (operation == SparseOperation.Nrm2) kernels.nrm2(x) else kernels.asum(x)
            }
        }
        "spnrm2-indexed" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val dense = Fixtures.vector(d[0], 2)
            arm(SparseOperation.IndexedNrm2, indices.size, "arithmetic") {
                kernels.nrm2(indices, 0, indices.size, dense)
            }
        }
        "spscatter-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            arm(SparseOperation.Scatter, x.values.size, "reset-and-arithmetic") {
                dense0.copyInto(dense)
                kernels.scatter(indices, 0, x.values, 0, x.values.size, dense)
                dense[0]
            }
        }
        "spgather" -> {
            val x = Fixtures.sparseVector(d[0], density, 1)
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            val timing = case.option("timing", "reset-and-arithmetic")
            arm(SparseOperation.Gather, x.values.size, timing, result = x.values) {
                if (timing == "reset-and-arithmetic") dense0.copyInto(dense)
                kernels.gather(x, dense)
                if (x.values.isEmpty()) 0.0 else x.values[0] + x.values[x.values.lastIndex]
            }
        }
        "spscatter", "spgather-zero" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val values0 = x.values.copyOf()
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            val scatter = case.operation == "spscatter"
            val operation = if (scatter) SparseOperation.Scatter else SparseOperation.GatherZero
            arm(operation, x.values.size, "reset-and-arithmetic") {
                values0.copyInto(x.values); dense0.copyInto(dense)
                if (scatter) kernels.scatter(x, dense) else kernels.gatherZero(x, dense)
                x.values.firstOrNull() ?: dense[0]
            }
        }
        else -> sparseMatrixArm(case, engine)
    }
}

/** The attribution a sparse row carries, taken from the binding rather than rebuilt from the mode string. */
internal fun sparseKernel(route: SparseRoute): String = "${route.implementation}/${route.entryPoint}"

/** A nonzero scatter epoch; the accumulator is cleared between iterations, so one value serves every pass. */
private const val EPOCH = 1

/**
 * Sparse Level 2 and 3 work, named by the route the call itself resolves.
 *
 * The scheduling is this library's portable CSC code on every engine, and the route says so; where a column is
 * handed to a Level 1 kernel, the route names the one that width reaches. An arm therefore never publishes a
 * portable sparse product under a SIMD label because the engine's Level 1 kernels happen to be vectorised.
 *
 * Every case verifies its own result before timing. A prepared row is checked against the one-shot call it is
 * meant to be a faster way of making, so a snapshot that quietly computed something else is a failure rather
 * than a fast number.
 */
@Suppress("LongMethod", "CyclomaticComplexMethod") // one branch per benchmarked operation
internal fun sparseMatrixArm(case: BenchCase, engine: KoblasEngine): ArmChoice? {
    val density = case.option("density", "0.01").toDouble()
    val d = case.dimensions
    val lower = case.option("uplo", "L") == "L"
    val transpose = case.flag("transA")
    val unit = case.option("diag", "U") == "U"
    val right = case.option("side", "L") == "R"
    val mode = case.option("mode", "oneshot")
    val alpha = 0.875
    val beta = -0.25
    val workspace = MatrixWorkspace()

    fun arm(operation: SparseMatrixOperation, entriesPerColumn: Int, denseRun: Int, timing: String, run: () -> Double): ArmChoice {
        val route = engine.matrixRouteOf(operation, entriesPerColumn, denseRun)
        if (!route.exactlyMeasurable) return ArmChoice(null, requireNotNull(route.reason))
        preflight(case, run)
        return ArmChoice(CaseWork("direct", timing, run, kernel = sparseMatrixKernel(route)), null)
    }

    return when (case.operation) {
        "spgemv" -> {
            val a = Fixtures.sparse(d[0], d[1], density, 1)
            val x = Fixtures.vector(d[1], 2)
            val y0 = Fixtures.vector(d[0], 3)
            val y = y0.copyOf()
            val operation = SparseMatrixOperation.Gemv
            preparedArm(case, engine, operation, storedPerColumn(a), 0, mode, a,
                oneShot = { y0.copyInto(y); engine.gemv(alpha, a, x, beta, y); y[0] },
                prepared = { prepared -> y0.copyInto(y); prepared.gemv(alpha, x, beta, y); y[0] },
            )
        }

        "spmm", "spmm-generic", "spmm-generic-right" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(m, k, density, 1)
            val b = Fixtures.matrix(k, n, 2)
            val c0 = Fixtures.matrix(m, n, 3)
            val c = Fixtures.matrix(m, n, 3)
            val mirrorB = Fixtures.matrix(n, m, 2)
            val mirror0 = Fixtures.matrix(n, k, 3)
            val mirror = Fixtures.matrix(n, k, 3)
            when (case.operation) {
                // The generic entry point, timed with its dispatch, for a sparse operand on each side.
                "spmm-generic" -> arm(SparseMatrixOperation.GemmDense, storedPerColumn(a), 0, "oneshot-generic") {
                    c0.values.copyInto(c.values)
                    (a as Matrix).gemmInto(alpha, false, b as Matrix, false, beta, c, workspace)
                    c.values[0]
                }

                "spmm-generic-right" -> arm(SparseMatrixOperation.GemmDenseRight, storedPerColumn(a), n, "oneshot-generic") {
                    mirror0.values.copyInto(mirror.values)
                    (mirrorB as Matrix).gemmInto(alpha, false, a as Matrix, false, beta, mirror, workspace)
                    mirror.values[0]
                }

                else -> preparedArm(case, engine, SparseMatrixOperation.GemmDense, storedPerColumn(a), 0, mode, a,
                    oneShot = {
                        c0.values.copyInto(c.values)
                        engine.gemm(alpha, a, false, b, false, beta, c, workspace = workspace)
                        c.values[0]
                    },
                    prepared = { prepared ->
                        c0.values.copyInto(c.values)
                        prepared.gemm(alpha, false, b, beta, c, workspace)
                        c.values[0]
                    },
                )
            }
        }

        "spgemm", "spgemm-generic" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(m, k, density, 1)
            val b = Fixtures.sparse(k, n, density, 2)
            if (case.operation == "spgemm-generic") {
                arm(SparseMatrixOperation.GemmSparse, storedPerColumn(a), 0, "oneshot-generic") {
                    ((a as Matrix) * (b as Matrix)).let { (it as SparseMatrix).values.firstOrNull() ?: 0.0 }
                }
            } else {
                preparedArm(case, engine, SparseMatrixOperation.GemmSparse, storedPerColumn(a), 0, mode, a,
                    oneShot = { engine.gemm(a, b).values.firstOrNull() ?: 0.0 },
                    prepared = { prepared -> prepared.gemm(b).values.firstOrNull() ?: 0.0 },
                )
            }
        }

        "spsymv" -> {
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower)
            val x = Fixtures.vector(d[0], 2)
            val y0 = Fixtures.vector(d[0], 3)
            val y = y0.copyOf()
            arm(SparseMatrixOperation.Symv, storedPerColumn(a), 0, "oneshot") {
                y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower); y[0]
            }
        }

        "spsymm" -> {
            val (n, rhs) = d
            val a = Fixtures.sparse(n, n, density, 1, triangular = true, lower = lower)
            val b = if (right) Fixtures.matrix(rhs, n, 2) else Fixtures.matrix(n, rhs, 2)
            val c0 = Fixtures.matrix(b.rows, b.cols, 3)
            val c = Fixtures.matrix(b.rows, b.cols, 3)
            arm(SparseMatrixOperation.Symm, storedPerColumn(a), 0, "oneshot") {
                c0.values.copyInto(c.values)
                engine.symm(alpha, a, b, beta, c, lower, right, workspace)
                c.values[0]
            }
        }

        "sptrsv", "sptrmv" -> {
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower)
            val x0 = Fixtures.vector(d[0], 2)
            val x = x0.copyOf()
            val solve = case.operation == "sptrsv"
            val operation = if (solve) SparseMatrixOperation.Trsv else SparseMatrixOperation.Trmv
            arm(operation, storedPerColumn(a), 0, "oneshot") {
                x0.copyInto(x)
                if (solve) engine.trsv(a, x, lower, transpose, unit) else engine.trmv(a, x, lower, transpose, unit)
                x[0]
            }
        }

        "sptrsm", "sptrmm" -> {
            val (order, rhs) = d
            val a = Fixtures.sparse(order, order, density, 1, triangular = true, lower = lower)
            val original = Fixtures.matrix(if (right) rhs else order, if (right) order else rhs, 2)
            val b = Fixtures.matrix(original.rows, original.cols, 2)
            val solve = case.operation == "sptrsm"
            val operation = when {
                solve && right -> SparseMatrixOperation.TrsmRight
                solve -> SparseMatrixOperation.TrsmLeft
                right -> SparseMatrixOperation.TrmmRight
                else -> SparseMatrixOperation.TrmmLeft
            }
            arm(operation, storedPerColumn(a), b.rows, "oneshot") {
                original.values.copyInto(b.values)
                if (solve) {
                    engine.trsm(a, b, lower, transpose, unit, right, alpha, workspace)
                } else {
                    engine.trmm(a, b, lower, transpose, unit, right, alpha, workspace)
                }
                b.values[0]
            }
        }

        "spsyrk-dense", "spsyrk-sparse" -> {
            val (n, k) = d
            val a = Fixtures.sparse(n, k, density, 1)
            val c0 = Fixtures.matrix(n, n, 2)
            val c = Fixtures.matrix(n, n, 2)
            val dense = case.operation == "spsyrk-dense"
            val operation = if (dense) SparseMatrixOperation.SyrkDense else SparseMatrixOperation.SyrkSparse
            arm(operation, storedPerColumn(a), 0, "oneshot") {
                if (dense) {
                    c0.values.copyInto(c.values)
                    engine.syrk(alpha, a, false, beta, c, lower, workspace)
                    c.values[0]
                } else {
                    engine.syrk(a, false, lower).values.firstOrNull() ?: 0.0
                }
            }
        }

        "spadd" -> {
            val (m, n) = d
            val a = Fixtures.sparse(m, n, density, 1)
            val b = Fixtures.sparse(m, n, density, 2)
            arm(SparseMatrixOperation.AddScaled, storedPerColumn(a), 0, "oneshot") {
                engine.addScaled(alpha, a, false, b).values.firstOrNull() ?: 0.0
            }
        }

        else -> null
    }
}

/**
 * The four accounting boundaries a prepared operand has.
 *
 * `oneshot` never builds a snapshot. `prepared` builds one outside the timed region and times reuse.
 * `setup` times building one alone, and `firstuse` times building one and calling it once, which is where a
 * derived orientation is paid for. Comparing across them is comparing different work, so they are separate
 * cases rather than one row with an option.
 */
@Suppress("LongParameterList") // the operation, its route inputs, the mode and both call shapes
private fun preparedArm(
    case: BenchCase,
    engine: KoblasEngine,
    operation: SparseMatrixOperation,
    entriesPerColumn: Int,
    denseRun: Int,
    mode: String,
    source: SparseMatrix,
    oneShot: () -> Double,
    prepared: (PreparedSparseMatrix) -> Double,
): ArmChoice {
    val route = engine.matrixRouteOf(operation, entriesPerColumn, denseRun)
    if (!route.exactlyMeasurable) return ArmChoice(null, requireNotNull(route.reason))
    val kernel = sparseMatrixKernel(route)
    val run: () -> Double = when (mode) {
        "oneshot" -> oneShot
        "setup" -> ({ engine.prepare(source).nnz.toDouble() })
        "firstuse" -> ({ prepared(engine.prepare(source)) })
        else -> {
            val snapshot = engine.prepare(source)
            // Checked against the one-shot call before anything is timed: a snapshot that computes something
            // else is a wrong answer, not a fast one.
            val expected = oneShot()
            val actual = prepared(snapshot)
            check(agree(expected, actual)) { "${case.id}: prepared result $actual disagrees with one-shot $expected" }
            ({ prepared(snapshot) })
        }
    }
    val timing = when (mode) {
        "oneshot" -> "oneshot"
        "setup" -> "prepare"
        "firstuse" -> "prepare-and-first-use"
        else -> "prepared"
    }
    preflight(case, run)
    return ArmChoice(CaseWork("direct", timing, run, kernel = kernel), null)
}

/** Runs a case once before it is timed, so a call that cannot produce a number never becomes a measurement. */
private fun preflight(case: BenchCase, run: () -> Double) {
    val value = run()
    check(!value.isNaN()) { "${case.id}: the case produced NaN before timing" }
}

/** Whether two case results are the same number, to the tolerance a different summation order leaves. */
private fun agree(expected: Double, actual: Double): Boolean {
    if (expected == actual) return true
    if (!expected.isFinite() || !actual.isFinite()) return false
    return abs(expected - actual) <= 1e-9 * maxOf(1.0, abs(expected))
}

/**
 * Stored entries per column of a fixture, which is what decides the Level 1 leaf a scattered column reaches.
 *
 * The uniform sparse fixture gives every column the same count, so one number answers for the whole call. A
 * triangular fixture does not, but its operations call no indexed leaf, so the number does not reach the route.
 */
private fun storedPerColumn(a: SparseMatrix): Int = if (a.cols == 0) 0 else a.nnz / a.cols

/** The attribution a sparse matrix row carries, taken from the route the call resolved. */
internal fun sparseMatrixKernel(route: SparseMatrixRoute): String = "${route.implementation}/${route.entryPoint}"
