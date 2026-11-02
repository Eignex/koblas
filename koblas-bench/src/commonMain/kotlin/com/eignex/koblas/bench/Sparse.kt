package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.Matrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.SparsePrimitives

/**
 * The sparse work for one case on [engine], or the reason this engine cannot measure it, or null when the case
 * is not a sparse one.
 *
 * The engine-dispatched arms are exact because each runs on the engine its arm pinned through
 * [BuiltinEngines]. A row records that selection, not the body a given width reached: a
 * Vector API selection still runs scalar loops below its crossover.
 *
 * The generic primitives are the exception, because no engine selects them. They are one implementation, timed
 * once as an explicit composed case rather than repeated under each engine label.
 */
@Suppress("LongMethod", "CyclomaticComplexMethod") // one branch per benchmarked operation
internal fun sparseArm(case: BenchCase, engine: KoblasEngine): ArmChoice? {
    if (!case.operation.startsWith("sp")) return null
    val density = case.option("density", "0.01").toDouble()
    val d = case.dimensions
    val kernels = engine

    fun arm(count: Int, timing: String, result: DoubleArray? = null, run: () -> Double): ArmChoice =
        if (count == 0) {
            declined()
        } else {
            ArmChoice(engineWork(engine, case.operation, timing, run, result), null)
        }

    return when (case.operation) {
        "spbuild-triplets", "spbuild-duplicates" -> {
            if (engine !== BuiltinEngines.scalar) {
                ArmChoice(null, "the sparse factory is portable construction, timed on the scalar arm")
            } else {
                ArmChoice(sparseFactoryWork(case), null)
            }
        }
        "spdot-view" -> vectorViewsArm(case, engine)
        "spdot-column", "spdot-column-raw" -> sparseColumnDotArm(case, engine)
        "spdot-slice", "spnrm2-slice", "spaxpy-slice", "spscatter-slice" -> sparseSlicesArm(case, engine)
        "spaccumulate" -> {
            // The generic primitives are one implementation for every engine, so this is timed once rather
            // than repeated under each label, where it would compare a selection against a copy of itself.
            if (engine !== BuiltinEngines.scalar) {
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
            arm(x.values.size, "arithmetic") { kernels.dot(x, y) }
        }
        "spdot-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.vector(d[0], 2)
            val indices = x.copyIndices()
            arm(x.values.size, "arithmetic") {
                kernels.dot(indices, 0, x.values, 0, x.values.size, y)
            }
        }
        "spdot-sparse" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.sparseVector(d[0], density, 2)
            arm(x.values.size, "arithmetic") { kernels.dot(x, y) }
        }
        "spaxpy" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            arm(x.values.size, "reset-and-arithmetic") {
                y0.copyInto(y); kernels.axpy(y, 0.875, x); y[0]
            }
        }
        "spaxpy-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            arm(x.values.size, "reset-and-arithmetic") {
                y0.copyInto(y)
                kernels.axpy(y, 0.875, indices, 0, x.values, 0, x.values.size)
                y[0]
            }
        }
        "spnrm2", "spasum" -> {
            val x = Fixtures.sparseVector(d[0], density, 1)
            val norm = case.operation == "spnrm2"
            arm(x.values.size, "arithmetic") {
                if (norm) kernels.nrm2(x) else kernels.asum(x)
            }
        }
        "spnrm2-indexed" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val dense = Fixtures.vector(d[0], 2)
            arm(indices.size, "arithmetic") {
                kernels.nrm2(indices, 0, indices.size, dense)
            }
        }
        "spscatter-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            arm(x.values.size, "reset-and-arithmetic") {
                dense0.copyInto(dense)
                kernels.scatter(indices, 0, x.values, 0, x.values.size, dense)
                dense[0]
            }
        }
        "spgather" -> {
            val x = Fixtures.sparseVector(d[0], density, 1)
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            val timing = case.option("timing", "reset-and-arithmetic")
            arm(x.values.size, timing, result = x.values) {
                if (timing == "reset-and-arithmetic") dense0.copyInto(dense)
                kernels.gather(x, dense)
                if (x.values.isEmpty()) 0.0 else x.values[0] + x.values[x.values.lastIndex]
            }
        }
        "spscatter", "spgather-zero" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val values0 = x.values.copyOf()
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            val scatter = case.operation == "spscatter"
            arm(x.values.size, "reset-and-arithmetic") {
                values0.copyInto(x.values); dense0.copyInto(dense)
                if (scatter) kernels.scatter(x, dense) else kernels.gatherZero(x, dense)
                x.values.firstOrNull() ?: dense[0]
            }
        }
        else -> sparseMatrixArm(case, engine)
    }
}

/** A nonzero scatter epoch; the accumulator is cleared between iterations, so one value serves every pass. */
private const val EPOCH = 1

/**
 * Sparse Level 2 and 3 work, published under the engine the arm pinned.
 *
 * The scheduling is this library's portable CSC code on every engine, so a row never shows that a sparse
 * product vectorised: the engine name records the Level 1 kernels a unit of work may be handed to, not which
 * ones a given width reached.
 *
 * Every case is verified against [SparseReference] before it is timed. The comparison covers the whole result
 * and, for a fresh CSC result, the support its operands' patterns reach; a case that computes the wrong thing
 * fails the run.
 */
@Suppress("LongMethod", "CyclomaticComplexMethod") // one branch per benchmarked operation
internal fun sparseMatrixArm(case: BenchCase, engine: KoblasEngine): ArmChoice? {
    val density = case.option("density", "0.01").toDouble()
    val support = case.option("support", "uniform")
    val d = case.dimensions
    val lower = case.option("uplo", "L") == "L"
    val transpose = case.flag("transA")
    val transposeDense = case.flag("transB")
    val unit = case.option("diag", "U") == "U"
    val right = case.option("side", "L") == "R"
    val alpha = 0.875
    val beta = -0.25
    val workspace = Workspace()

    fun arm(verify: () -> Unit, run: () -> Double) = publish(engine, case, "oneshot", verify, run)

    return when (case.operation) {
        "spgemv" -> {
            val a = Fixtures.sparse(d[0], d[1], density, 1, support = support)
            val reference = a.toArray()
            val x = Fixtures.vector(if (transpose) d[0] else d[1], 2)
            val y0 = Fixtures.vector(if (transpose) d[1] else d[0], 3)
            val y = y0.copyOf()
            val expected = SparseReference.gemv(alpha, reference, transpose, x, beta, y0)
            arm(
                verify = {
                    y0.copyInto(y)
                    engine.gemv(alpha, a, x, beta, y, transpose)
                    SparseReference.check(expected, y, "${case.id} gemv")
                },
            ) { y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, transpose); y[0] }
        }

        "spmm", "spmm-generic" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(if (transpose) k else m, if (transpose) m else k, density, 1, support = support)
            val reference = a.toArray()
            // Transposed storage puts the right-hand sides in adjacent rows. The seeded fixture is
            // reshaped for this layout; the reference uses these same stored values and transpose flag.
            val b = if (transposeDense) Fixtures.matrix(n, k, 2) else Fixtures.matrix(k, n, 2)
            val c0 = Fixtures.matrix(m, n, 3)
            val c = Fixtures.matrix(m, n, 3)
            val expected = SparseReference.gemm(
                alpha, reference, transpose, SparseReference.dense(b), transposeDense, beta,
                SparseReference.dense(c0),
            )
            if (case.operation == "spmm-generic") {
                genericArm(
                    case, engine,
                    verify = {
                        c0.values.copyInto(c.values)
                        engine.gemm(alpha, a as Matrix, transpose, b as Matrix, transposeDense, beta, c, workspace)
                        SparseReference.check(expected, SparseReference.dense(c), "${case.id} generic product")
                    },
                ) {
                    c0.values.copyInto(c.values)
                    engine.gemm(alpha, a as Matrix, transpose, b as Matrix, transposeDense, beta, c, workspace)
                    c.values[0]
                }
            } else {
                arm(
                    verify = {
                        c0.values.copyInto(c.values)
                        engine.gemm(alpha, a, transpose, b, transposeDense, beta, c, workspace = workspace)
                        SparseReference.check(expected, SparseReference.dense(c), "${case.id} product")
                    },
                ) {
                    c0.values.copyInto(c.values)
                    engine.gemm(alpha, a, transpose, b, transposeDense, beta, c, workspace = workspace)
                    c.values[0]
                }
            }
        }

        // The mirror of spmm on an exact arm: a dense operand on the left of the sparse one, through the
        // engine the row is published under rather than through the generic entry point, which has no
        // engine to be told and is a default-policy row.
        "spmm-right" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(
                if (transpose) k else m, if (transpose) m else k, density, 1, support = support,
            )
            val reference = a.toArray()
            val leftDense = Fixtures.matrix(n, m, 2)
            val mirror0 = Fixtures.matrix(n, k, 3)
            val mirror = Fixtures.matrix(n, k, 3)
            val expected = SparseReference.gemm(
                alpha, SparseReference.dense(leftDense), false, reference, transpose,
                beta, SparseReference.dense(mirror0),
            )
            arm(
                verify = {
                    mirror0.values.copyInto(mirror.values)
                    engine.gemm(
                        alpha,
                        a,
                        transpose,
                        leftDense,
                        false,
                        beta,
                        mirror,
                        right = true,
                        workspace = workspace,
                    )
                    SparseReference.check(expected, SparseReference.dense(mirror), "${case.id} product")
                },
            ) {
                mirror0.values.copyInto(mirror.values)
                engine.gemm(alpha, a, transpose, leftDense, false, beta, mirror, right = true, workspace = workspace)
                mirror.values[0]
            }
        }

        // The mirror of spmm: a dense operand on the left of the sparse one, through the same generic call.
        // op(A) is m by k whichever way the sparse operand is stored, so the dense operand is n by m and the
        // destination n by k.
        "spmm-generic-right" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(if (transpose) k else m, if (transpose) m else k, density, 1, support = support)
            val reference = a.toArray()
            val leftDense = Fixtures.matrix(n, m, 2)
            val mirror0 = Fixtures.matrix(n, k, 3)
            val mirror = Fixtures.matrix(n, k, 3)
            val expected = SparseReference.gemm(
                alpha, SparseReference.dense(leftDense), false, reference, transpose,
                beta, SparseReference.dense(mirror0),
            )
            genericArm(
                case, engine,
                verify = {
                    mirror0.values.copyInto(mirror.values)
                    engine.gemm(alpha, leftDense as Matrix, false, a as Matrix, transpose, beta, mirror, workspace)
                    SparseReference.check(expected, SparseReference.dense(mirror), "${case.id} generic product")
                },
            ) {
                mirror0.values.copyInto(mirror.values)
                engine.gemm(alpha, leftDense as Matrix, false, a as Matrix, transpose, beta, mirror, workspace)
                mirror.values[0]
            }
        }

        "spgemm", "spgemm-generic" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(if (transpose) k else m, if (transpose) m else k, density, 1, support = support)
            val b = Fixtures.sparse(k, n, density, 2, support = support)
            val expected = SparseReference.gemm(
                1.0, a.toArray(), transpose, b.toArray(), false, 0.0, Array(m) { DoubleArray(n) },
            )
            val support = SparseReference.productSupport(a, transpose, b, false)
            if (case.operation == "spgemm-generic") {
                genericArm(
                    case, engine,
                    verify = {
                        val product = engine.gemm(1.0, a as Matrix, transpose, b as Matrix, false)
                        SparseReference.checkSparse(
                            expected, support, product as SparseMatrix, "${case.id} generic product",
                        )
                    },
                ) {
                    Retained.retain(engine.gemm(1.0, a as Matrix, transpose, b as Matrix, false))
                }
            } else {
                arm(
                    verify = {
                        SparseReference.checkSparse(
                            expected, support, engine.gemm(1.0, a, transpose, b, false), "${case.id} product",
                        )
                    },
                ) { Retained.retain(engine.gemm(1.0, a, transpose, b, false)) }
            }
        }

        "spsymv" -> {
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower, support = support)
            val x = Fixtures.vector(d[0], 2)
            val y0 = Fixtures.vector(d[0], 3)
            val y = y0.copyOf()
            val expected = SparseReference.gemv(
                alpha, SparseReference.mirrored(a.toArray(), lower), false, x, beta, y0,
            )
            arm(
                verify = {
                    y0.copyInto(y)
                    engine.symv(alpha, a, x, beta, y, lower)
                    SparseReference.check(expected, y, "${case.id} symmetric product")
                },
            ) { y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower); y[0] }
        }

        "spsymm" -> {
            val (n, rhs) = d
            val a = Fixtures.sparse(n, n, density, 1, triangular = true, lower = lower, support = support)
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
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower, support = support)
            val x0 = Fixtures.vector(d[0], 2)
            val x = x0.copyOf()
            val solve = case.operation == "sptrsv"
            val triangle = SparseReference.triangle(a.toArray(), lower, unit)
            val expected = if (solve) {
                SparseReference.trsv(triangle, lower, transpose, x0)
            } else {
                SparseReference.trmv(triangle, transpose, x0)
            }
            arm(
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
            val a = Fixtures.sparse(order, order, density, 1, triangular = true, lower = lower, support = support)
            val original = Fixtures.matrix(if (right) rhs else order, if (right) order else rhs, 2)
            val b = Fixtures.matrix(original.rows, original.cols, 2)
            val solve = case.operation == "sptrsm"
            val triangle = SparseReference.triangle(a.toArray(), lower, unit)
            val expected = if (solve) {
                SparseReference.trsm(triangle, lower, transpose, right, alpha, SparseReference.dense(original))
            } else {
                SparseReference.trmm(triangle, transpose, right, alpha, SparseReference.dense(original))
            }
            arm(
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
            val a = Fixtures.sparse(n, k, density, 1, support = support)
            val c0 = Fixtures.matrix(n, n, 2)
            val c = Fixtures.matrix(n, n, 2)
            val dense = case.operation == "spsyrk-dense"
            val reference = a.toArray()
            arm(
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
                    Retained.retain(engine.syrk(a, false, lower))
                }
            }
        }

        "spadd" -> {
            val (m, n) = d
            val a = Fixtures.sparse(m, n, density, 1, support = support)
            val b = Fixtures.sparse(m, n, density, 2, support = support)
            val expected = SparseReference.addScaled(alpha, a.toArray(), false, b.toArray())
            val support = SparseReference.unionSupport(a, transposeA = false, b = b)
            arm(
                verify = {
                    SparseReference.checkSparse(
                        expected, support, engine.addScaled(alpha, a, false, b), "${case.id} sum",
                    )
                },
            ) { Retained.retain(engine.addScaled(alpha, a, false, b)) }
        }

        else -> null
    }
}


/** A timed row for a sparse matrix call on [engine], refused when one of the case's extents is empty. */
private fun publish(
    engine: KoblasEngine,
    case: BenchCase,
    timing: String,
    verify: () -> Unit,
    run: () -> Double,
): ArmChoice {
    if (case.dimensions.any { it == 0 }) return declined()
    verify()
    return ArmChoice(engineWork(engine, case.operation, timing, run), null)
}

/**
 * A row for the common `Matrix` product, which is a default-policy measurement rather than an exact arm.
 *
 * The generic entry point uses the platform-selected engine, so its timing mode keeps it apart from the
 * exact rows of the same operation.
 */
private fun genericArm(
    case: BenchCase,
    engine: KoblasEngine,
    verify: () -> Unit,
    run: () -> Double,
): ArmChoice = publish(engine, case, "oneshot-generic", verify, run)
