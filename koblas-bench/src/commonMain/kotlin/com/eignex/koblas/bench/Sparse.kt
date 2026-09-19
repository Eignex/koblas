package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Matrix
import com.eignex.koblas.MatrixWorkspace
import com.eignex.koblas.PreparedSparseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.gemmInto
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.SparseCall
import com.eignex.koblas.sparse.SparseMatrixOperation
import com.eignex.koblas.sparse.SparseMatrixRoute
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.sparse.SparsePrimitives
import com.eignex.koblas.sparse.SparseRoute
import com.eignex.koblas.times
import com.eignex.koblas.vendor.RouteKind
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
 * The scheduling is this library's portable CSC code on every engine, and the route says so; where a unit of
 * work is handed to a Level 1 kernel, the route names the one that unit reaches. An arm therefore never
 * publishes a portable sparse product under a SIMD label because the engine's Level 1 kernels are vectorised.
 * A route the call cannot make exact is published as the composition it is rather than under either name.
 *
 * Every case is verified against [SparseReference] before it is timed, and a prepared case is verified through
 * the snapshot as well as through the one-shot call. The comparison covers the whole result and, for a fresh
 * CSC result, the support its operands' patterns reach; a case that computes the wrong thing fails the run.
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

    fun arm(operation: SparseMatrixOperation, call: SparseCall, verify: () -> Unit, run: () -> Double) =
        publish(engine, operation, call, "oneshot", verify, run)

    return when (case.operation) {
        "spgemv" -> {
            val a = Fixtures.sparse(d[0], d[1], density, 1)
            val reference = a.toArray()
            val x = Fixtures.vector(if (transpose) d[0] else d[1], 2)
            val y0 = Fixtures.vector(if (transpose) d[1] else d[0], 3)
            val y = y0.copyOf()
            val expected = SparseReference.gemv(alpha, reference, transpose, x, beta, y0)
            val operation = if (transpose) SparseMatrixOperation.GemvTransposed else SparseMatrixOperation.Gemv
            val call = SparseCall(
                a, alpha, beta,
                destinationElements = y.size,
                depth = SparseReference.columns(reference, transpose),
            )
            preparedArm(
                case, engine, operation, call, mode, a,
                verifyOneShot = {
                    y0.copyInto(y)
                    engine.gemv(alpha, a, x, beta, y, transpose)
                    SparseReference.check(expected, y, "${case.id} gemv")
                },
                verifyPrepared = { snapshot ->
                    y0.copyInto(y)
                    snapshot.gemv(alpha, x, beta, y, transpose)
                    SparseReference.check(expected, y, "${case.id} prepared gemv")
                },
                oneShot = { y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, transpose); y[0] },
                prepared = { snapshot -> y0.copyInto(y); snapshot.gemv(alpha, x, beta, y, transpose); y[0] },
            )
        }

        "spmm", "spmm-generic" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(if (transpose) k else m, if (transpose) m else k, density, 1)
            val reference = a.toArray()
            val b = Fixtures.matrix(k, n, 2)
            val c0 = Fixtures.matrix(m, n, 3)
            val c = Fixtures.matrix(m, n, 3)
            val expected = SparseReference.gemm(
                alpha, reference, transpose, SparseReference.dense(b), false, beta, SparseReference.dense(c0),
            )
            val call = SparseCall(a, alpha, beta, destinationElements = c.values.size, depth = k)
            if (case.operation == "spmm-generic") {
                genericArm(
                    case, engine, SparseMatrixOperation.GemmDense, call,
                    verify = {
                        c0.values.copyInto(c.values)
                        (a as Matrix).gemmInto(alpha, transpose, b as Matrix, false, beta, c, workspace)
                        SparseReference.check(expected, SparseReference.dense(c), "${case.id} generic product")
                    },
                ) {
                    c0.values.copyInto(c.values)
                    (a as Matrix).gemmInto(alpha, transpose, b as Matrix, false, beta, c, workspace)
                    c.values[0]
                }
            } else {
                preparedArm(
                    case, engine, SparseMatrixOperation.GemmDense, call, mode, a,
                    verifyOneShot = {
                        c0.values.copyInto(c.values)
                        engine.gemm(alpha, a, transpose, b, false, beta, c, workspace = workspace)
                        SparseReference.check(expected, SparseReference.dense(c), "${case.id} product")
                    },
                    verifyPrepared = { snapshot ->
                        c0.values.copyInto(c.values)
                        snapshot.gemm(alpha, transpose, b, beta, c, workspace)
                        SparseReference.check(expected, SparseReference.dense(c), "${case.id} prepared product")
                    },
                    oneShot = {
                        c0.values.copyInto(c.values)
                        engine.gemm(alpha, a, transpose, b, false, beta, c, workspace = workspace)
                        c.values[0]
                    },
                    prepared = { snapshot ->
                        c0.values.copyInto(c.values)
                        snapshot.gemm(alpha, transpose, b, beta, c, workspace)
                        c.values[0]
                    },
                )
            }
        }

        // The mirror of spmm: a dense operand on the left of the sparse one, through the same generic call.
        // op(A) is m by k whichever way the sparse operand is stored, so the dense operand is n by m and the
        // destination n by k.
        "spmm-generic-right" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(if (transpose) k else m, if (transpose) m else k, density, 1)
            val reference = a.toArray()
            val leftDense = Fixtures.matrix(n, m, 2)
            val mirror0 = Fixtures.matrix(n, k, 3)
            val mirror = Fixtures.matrix(n, k, 3)
            val expected = SparseReference.gemm(
                alpha, SparseReference.dense(leftDense), false, reference, transpose,
                beta, SparseReference.dense(mirror0),
            )
            genericArm(
                case, engine, SparseMatrixOperation.GemmDenseRight,
                SparseCall(
                    a, alpha, beta,
                    destinationElements = mirror.values.size, depth = m, updateRun = mirror.rows,
                ),
                verify = {
                    mirror0.values.copyInto(mirror.values)
                    (leftDense as Matrix).gemmInto(alpha, false, a as Matrix, transpose, beta, mirror, workspace)
                    SparseReference.check(expected, SparseReference.dense(mirror), "${case.id} generic product")
                },
            ) {
                mirror0.values.copyInto(mirror.values)
                (leftDense as Matrix).gemmInto(alpha, false, a as Matrix, transpose, beta, mirror, workspace)
                mirror.values[0]
            }
        }

        "spgemm", "spgemm-generic" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(if (transpose) k else m, if (transpose) m else k, density, 1)
            val b = Fixtures.sparse(k, n, density, 2)
            val expected = SparseReference.gemm(
                1.0, a.toArray(), transpose, b.toArray(), false, 0.0, Array(m) { DoubleArray(n) },
            )
            val support = SparseReference.productSupport(a, transpose, b, false)
            val call = SparseCall(a, depth = k)
            if (case.operation == "spgemm-generic") {
                genericArm(
                    case, engine, SparseMatrixOperation.GemmSparse, call,
                    verify = {
                        val product = (a as Matrix) * (b as Matrix)
                        SparseReference.checkSparse(
                            expected, support, product as SparseMatrix, "${case.id} generic product",
                        )
                    },
                ) {
                    ((a as Matrix) * (b as Matrix)).let { (it as SparseMatrix).values.firstOrNull() ?: 0.0 }
                }
            } else {
                preparedArm(
                    case, engine, SparseMatrixOperation.GemmSparse, call, mode, a,
                    verifyOneShot = {
                        SparseReference.checkSparse(
                            expected, support, engine.gemm(1.0, a, transpose, b, false), "${case.id} product",
                        )
                    },
                    verifyPrepared = { snapshot ->
                        SparseReference.checkSparse(
                            expected, support, snapshot.gemm(1.0, transpose, b, false),
                            "${case.id} prepared product",
                        )
                    },
                    oneShot = { engine.gemm(1.0, a, transpose, b, false).values.firstOrNull() ?: 0.0 },
                    prepared = { snapshot -> snapshot.gemm(1.0, transpose, b, false).values.firstOrNull() ?: 0.0 },
                )
            }
        }

        "spsymv" -> {
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower)
            val x = Fixtures.vector(d[0], 2)
            val y0 = Fixtures.vector(d[0], 3)
            val y = y0.copyOf()
            val expected = SparseReference.gemv(
                alpha, SparseReference.mirrored(a.toArray(), lower), false, x, beta, y0,
            )
            arm(
                SparseMatrixOperation.Symv,
                SparseCall(a, alpha, beta, destinationElements = y.size, depth = d[0]),
                verify = {
                    y0.copyInto(y)
                    engine.symv(alpha, a, x, beta, y, lower)
                    SparseReference.check(expected, y, "${case.id} symmetric product")
                },
            ) { y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower); y[0] }
        }

        "spsymm" -> {
            val (n, rhs) = d
            val a = Fixtures.sparse(n, n, density, 1, triangular = true, lower = lower)
            val b = if (right) Fixtures.matrix(rhs, n, 2) else Fixtures.matrix(n, rhs, 2)
            val c0 = Fixtures.matrix(b.rows, b.cols, 3)
            val c = Fixtures.matrix(b.rows, b.cols, 3)
            val full = SparseReference.mirrored(a.toArray(), lower)
            val expected = if (right) {
                SparseReference.gemm(
                    alpha, SparseReference.dense(b), false, full, false, beta, SparseReference.dense(c0),
                )
            } else {
                SparseReference.gemm(
                    alpha, full, false, SparseReference.dense(b), false, beta, SparseReference.dense(c0),
                )
            }
            arm(
                SparseMatrixOperation.Symm,
                SparseCall(a, alpha, beta, destinationElements = c.values.size, depth = n),
                verify = {
                    c0.values.copyInto(c.values)
                    engine.symm(alpha, a, b, beta, c, lower, right, workspace)
                    SparseReference.check(expected, SparseReference.dense(c), "${case.id} symmetric product")
                },
            ) {
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
            val triangle = SparseReference.triangle(a.toArray(), lower, unit)
            val expected = if (solve) {
                SparseReference.trsv(triangle, lower, transpose, x0)
            } else {
                SparseReference.trmv(triangle, transpose, x0)
            }
            val operation = if (solve) SparseMatrixOperation.Trsv else SparseMatrixOperation.Trmv
            arm(
                operation, SparseCall(a, destinationElements = x.size),
                verify = {
                    x0.copyInto(x)
                    if (solve) engine.trsv(a, x, lower, transpose, unit) else engine.trmv(a, x, lower, transpose, unit)
                    SparseReference.check(expected, x, "${case.id} triangular")
                },
            ) {
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
            val triangle = SparseReference.triangle(a.toArray(), lower, unit)
            val expected = if (solve) {
                SparseReference.trsm(triangle, lower, transpose, right, alpha, SparseReference.dense(original))
            } else {
                SparseReference.trmm(triangle, transpose, right, alpha, SparseReference.dense(original))
            }
            val operation = when {
                solve && right -> SparseMatrixOperation.TrsmRight
                solve -> SparseMatrixOperation.TrsmLeft
                right -> SparseMatrixOperation.TrmmRight
                else -> SparseMatrixOperation.TrmmLeft
            }
            arm(
                operation,
                SparseCall(a, alpha, destinationElements = b.values.size, updateRun = b.rows),
                verify = {
                    original.values.copyInto(b.values)
                    if (solve) {
                        engine.trsm(a, b, lower, transpose, unit, right, alpha, workspace)
                    } else {
                        engine.trmm(a, b, lower, transpose, unit, right, alpha, workspace)
                    }
                    SparseReference.check(expected, SparseReference.dense(b), "${case.id} triangular block")
                },
            ) {
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
            val reference = a.toArray()
            val operation = if (dense) SparseMatrixOperation.SyrkDense else SparseMatrixOperation.SyrkSparse
            val call = SparseCall(
                a, alpha, beta,
                destinationElements = if (dense) c.values.size else null,
                depth = k,
            )
            arm(
                operation, call,
                verify = {
                    if (dense) {
                        c0.values.copyInto(c.values)
                        engine.syrk(alpha, a, false, beta, c, lower, workspace)
                        SparseReference.check(
                            SparseReference.syrk(alpha, reference, false, beta, SparseReference.dense(c0), lower),
                            SparseReference.dense(c), "${case.id} rank update",
                        )
                    } else {
                        val zero = Array(n) { DoubleArray(n) }
                        SparseReference.checkSparse(
                            SparseReference.syrk(1.0, reference, false, 0.0, zero, lower),
                            SparseReference.rankSupport(a, transpose = false, lower = lower),
                            engine.syrk(a, false, lower), "${case.id} rank update",
                        )
                    }
                },
            ) {
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
            val expected = SparseReference.addScaled(alpha, a.toArray(), false, b.toArray())
            val support = SparseReference.unionSupport(a, transposeA = false, b = b)
            arm(
                SparseMatrixOperation.AddScaled, SparseCall(a, alpha),
                verify = {
                    SparseReference.checkSparse(
                        expected, support, engine.addScaled(alpha, a, false, b), "${case.id} sum",
                    )
                },
            ) { engine.addScaled(alpha, a, false, b).values.firstOrNull() ?: 0.0 }
        }

        else -> null
    }
}

/**
 * A timed row for a sparse matrix call, refused when the call would do no work at all.
 *
 * A composed route is published rather than declined: the sparse scheduling that owns the call really did run
 * on this engine, and naming every component it can reach is the honest report. What a row may not do is
 * claim one implementation for units of work that reached another, which is what the route's kind records.
 */
private fun publish(
    engine: KoblasEngine,
    operation: SparseMatrixOperation,
    call: SparseCall,
    timing: String,
    verify: () -> Unit,
    run: () -> Double,
): ArmChoice {
    val route = engine.matrixRouteOf(operation, call)
    if (route.kind == RouteKind.NoWork) return declined()
    verify()
    return ArmChoice(CaseWork(route.kind.name.lowercase(), timing, run, kernel = sparseMatrixKernel(route)), null)
}

private fun declined(): ArmChoice =
    ArmChoice(null, "this case's own contract stops before the arithmetic, so there is nothing to time")

/**
 * A row for the common `Matrix` product, which is a default-policy measurement rather than an exact arm.
 *
 * The generic entry point is the user-facing one, and it uses the engine this platform selected rather than
 * one a benchmark names: a caller holding a `Matrix` has no engine to pass. Timing it under an arm whose
 * engine is a different one would publish that engine's label over another engine's work, so the case runs
 * only on the arm whose engine is the selected one and reports the route that engine resolves.
 */
private fun genericArm(
    case: BenchCase,
    engine: KoblasEngine,
    operation: SparseMatrixOperation,
    call: SparseCall,
    verify: () -> Unit,
    run: () -> Double,
): ArmChoice {
    if (engine !== koblas) {
        return ArmChoice(
            null,
            "the common Matrix product uses the platform-selected engine, so ${case.operation} is timed once " +
                "as a default-policy case on the arm whose engine that is",
        )
    }
    val route = koblas.matrixRouteOf(operation, call)
    if (route.kind == RouteKind.NoWork) return declined()
    verify()
    return ArmChoice(
        CaseWork("default-policy", "oneshot-generic", run, kernel = sparseMatrixKernel(route)),
        null,
    )
}

/**
 * The four accounting boundaries a prepared operand has.
 *
 * `oneshot` never builds a snapshot. `prepared` builds one outside the timed region and times reuse. `setup`
 * times building one alone, and reports snapshot preparation rather than the arithmetic kernel of a call that
 * did not happen. `firstuse` times building one and calling it once, which is where a derived orientation is
 * paid for, and reports the composition of the two. Comparing across them is comparing different work, so
 * they are separate cases rather than one row with an option.
 *
 * Both paths are checked against the reference before anything is timed, and the prepared check runs against
 * a snapshot that has not been used yet, so a cold transposed first use is covered rather than assumed.
 */
@Suppress("LongParameterList") // the operation, its route, the mode, and both verified call shapes
private fun preparedArm(
    case: BenchCase,
    engine: KoblasEngine,
    operation: SparseMatrixOperation,
    call: SparseCall,
    mode: String,
    source: SparseMatrix,
    verifyOneShot: () -> Unit,
    verifyPrepared: (PreparedSparseMatrix) -> Unit,
    oneShot: () -> Double,
    prepared: (PreparedSparseMatrix) -> Double,
): ArmChoice {
    if (mode == "setup") {
        val route = engine.matrixRouteOf(SparseMatrixOperation.Prepare, SparseCall(source))
        return ArmChoice(
            CaseWork(
                route.kind.name.lowercase(), "prepare", { engine.prepare(source).nnz.toDouble() },
                kernel = sparseMatrixKernel(route),
            ),
            null,
        )
    }
    val route = engine.matrixRouteOf(operation, call)
    if (route.kind == RouteKind.NoWork) return declined()
    verifyOneShot()
    if (mode == "oneshot") {
        return ArmChoice(CaseWork(route.kind.name.lowercase(), mode, oneShot, kernel = sparseMatrixKernel(route)), null)
    }
    // A snapshot that has done nothing yet, so the check covers the orientation a transposed call derives on
    // its first use rather than a warm one.
    verifyPrepared(engine.prepare(source))
    if (mode == "firstuse") {
        val prepare = engine.matrixRouteOf(SparseMatrixOperation.Prepare, SparseCall(source))
        val kernel = "${prepare.implementation}/${prepare.entryPoint} then ${sparseMatrixKernel(route)}"
        return ArmChoice(
            CaseWork("composed", "prepare-and-first-use", { prepared(engine.prepare(source)) }, kernel = kernel),
            null,
        )
    }
    val snapshot = engine.prepare(source)
    prepared(snapshot)
    return ArmChoice(
        CaseWork(route.kind.name.lowercase(), mode, { prepared(snapshot) }, kernel = sparseMatrixKernel(route)),
        null,
    )
}

/** The attribution a sparse matrix row carries, taken from the route the call resolved. */
internal fun sparseMatrixKernel(route: SparseMatrixRoute): String = "${route.implementation}/${route.entryPoint}"
