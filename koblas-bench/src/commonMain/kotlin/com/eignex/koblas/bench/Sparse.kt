@file:OptIn(com.eignex.koblas.UnsafeKoblasApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.operations.*

/**
 * The sparse work for one case on [engine], or the reason this engine cannot measure it, or null when the case
 * is not a sparse one.
 *
 * The engine-dispatched arms are exact because each runs on the engine its arm pinned through
 * [KoblasEngine]. A row records that selection, not the body a given width reached: a
 * Vector API selection still runs scalar loops below its crossover.
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
            if (engine !== KoblasEngine.scalar) {
                ArmChoice(null, "the sparse factory is portable construction, timed on the scalar arm")
            } else {
                ArmChoice(sparseFactoryWork(case), null)
            }
        }
        "spdot-view" -> vectorViewsArm(case, engine)
        "spdot-column", "spdot-column-raw" -> sparseColumnDotArm(case, engine)
        "spdot-slice", "spaxpy-slice", "spscatter-slice" -> sparseSlicesArm(case, engine)
        "spdot" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.vector(d[0], 2)
            arm(x.values.size, "arithmetic") { kernels.dot(x.indices, 0, x.values, 0, x.values.size, y) }
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
            arm(x.values.size, "arithmetic") { kernels.dot(x.indices, 0, x.values, 0, x.values.size, y.indices, 0, y.values, 0, y.values.size) }
        }
        "spaxpy" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            arm(x.values.size, "reset-and-arithmetic") {
                y0.copyInto(y); kernels.axpy(x.indices, 0, x.values, 0, x.values.size, 0.875, y); y[0]
            }
        }
        "spaxpy-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            arm(x.values.size, "reset-and-arithmetic") {
                y0.copyInto(y)
                kernels.axpy(indices, 0, x.values, 0, x.values.size, 0.875, y)
                y[0]
            }
        }
        "spnrm2", "spasum" -> {
            val x = Fixtures.sparseVector(d[0], density, 1)
            val norm = case.operation == "spnrm2"
            arm(x.values.size, "arithmetic") {
                if (norm) kernels.normVector(x) else kernels.absoluteSumVector(x)
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
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            val timing = case.option("timing", "reset-and-arithmetic")
            arm(x.values.size, timing, result = x.values) {
                if (timing == "reset-and-arithmetic") dense0.copyInto(dense)
                kernels.gather(indices, 0, x.values, 0, x.values.size, dense)
                if (x.values.isEmpty()) 0.0 else x.values[0] + x.values[x.values.lastIndex]
            }
        }
        "spscatter", "spgather-zero" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val values0 = x.values.copyOf()
            val indices = x.copyIndices()
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            val scatter = case.operation == "spscatter"
            arm(x.values.size, "reset-and-arithmetic") {
                values0.copyInto(x.values); dense0.copyInto(dense)
                if (scatter) kernels.scatter(x.indices, 0, x.values, 0, x.values.size, dense) else kernels.gatherZero(indices, 0, x.values, 0, x.values.size, dense)
                x.values.firstOrNull() ?: dense[0]
            }
        }
        else -> sparseMatrixArm(case, engine)
    }
}

/**
 * Sparse Level 2 and 3 work, published under the engine the arm pinned.
 *
 * The scheduling is this library's portable CSC code on every engine, so a row never shows that a sparse
 * product vectorised: the engine name records the Level 1 kernels a unit of work may be handed to, not which
 * ones a given width reached.
 *
 * Every case is verified against [SparseReference] before it is timed. The comparison covers the whole result
 * and a case that computes the wrong thing fails the run.
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
    val scratch = Scratch()

    fun arm(verify: () -> Unit, run: () -> Double) = publish(engine, case, "oneshot", verify, run)

    return when (case.operation) {
        "spgemv" -> {
            val a = Fixtures.sparse(d[0], d[1], density, 1, support = support)
            val x = Fixtures.vector(if (transpose) d[0] else d[1], 2)
            val y0 = Fixtures.vector(if (transpose) d[1] else d[0], 3)
            val y = y0.copyOf()
            val expected = DenseReference.gemv(alpha, SparseReference.dense(a), transpose, x, beta, y0)
            arm(
                verify = {
                    y0.copyInto(y)
                    engine.gemv(alpha, a, x, beta, y, transpose)
                    DenseReference.check(expected, y, "${case.id} gemv")
                },
            ) { y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, transpose); y[0] }
        }

        "spmm" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(if (transpose) k else m, if (transpose) m else k, density, 1, support = support)
            // Transposed storage puts the right-hand sides in adjacent rows. The seeded fixture is
            // reshaped for this layout; the reference uses these same stored values and transpose flag.
            val b = if (transposeDense) Fixtures.matrix(n, k, 2) else Fixtures.matrix(k, n, 2)
            val c0 = Fixtures.matrix(m, n, 3)
            val c = Fixtures.matrix(m, n, 3)
            val expected = DenseReference.gemm(alpha, SparseReference.dense(a), transpose, b, transposeDense, beta, c0)
            arm(
                verify = {
                    c0.values.copyInto(c.values)
                    engine.gemm(alpha, a, transpose, b, transposeDense, beta, c, scratch = scratch)
                    DenseReference.check(expected, c.values, "${case.id} product")
                },
            ) {
                c0.values.copyInto(c.values)
                engine.gemm(alpha, a, transpose, b, transposeDense, beta, c, scratch = scratch)
                c.values[0]
            }

        }

        // The right-side form multiplies a dense operand by the sparse one on the selected engine.
        "spmm-right" -> {
            val (m, n, k) = d
            val a = Fixtures.sparse(
                if (transpose) k else m, if (transpose) m else k, density, 1, support = support,
            )
            val leftDense = Fixtures.matrix(n, m, 2)
            val mirror0 = Fixtures.matrix(n, k, 3)
            val mirror = Fixtures.matrix(n, k, 3)
            val expected =
                DenseReference.gemm(alpha, leftDense, false, SparseReference.dense(a), transpose, beta, mirror0)
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
                        scratch = scratch,
                    )
                    DenseReference.check(expected, mirror.values, "${case.id} product")
                },
            ) {
                mirror0.values.copyInto(mirror.values)
                engine.gemm(alpha, a, transpose, leftDense, false, beta, mirror, right = true, scratch = scratch)
                mirror.values[0]
            }
        }

        "spsymv" -> {
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower, support = support)
            val x = Fixtures.vector(d[0], 2)
            val y0 = Fixtures.vector(d[0], 3)
            val y = y0.copyOf()
            val expected = DenseReference.symv(alpha, SparseReference.dense(a), lower, x, beta, y0)
            arm(
                verify = {
                    y0.copyInto(y)
                    engine.symv(alpha, a, x, beta, y, lower)
                    DenseReference.check(expected, y, "${case.id} symmetric product")
                },
            ) { y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower); y[0] }
        }

        "spsymm" -> {
            val (n, rhs) = d
            val a = Fixtures.sparse(n, n, density, 1, triangular = true, lower = lower, support = support)
            val b = if (right) Fixtures.matrix(rhs, n, 2) else Fixtures.matrix(n, rhs, 2)
            val c0 = Fixtures.matrix(b.rows, b.cols, 3)
            val c = Fixtures.matrix(b.rows, b.cols, 3)
            val expected = DenseReference.symm(alpha, SparseReference.dense(a), lower, right, b, beta, c0)
            arm(
                verify = {
                    c0.values.copyInto(c.values)
                    engine.symm(alpha, a, b, beta, c, lower, right, scratch = scratch)
                    DenseReference.check(expected, c.values, "${case.id} symmetric product")
                },
            ) {
                c0.values.copyInto(c.values)
                engine.symm(alpha, a, b, beta, c, lower, right, scratch = scratch)
                c.values[0]
            }
        }

        "sptrsv", "sptrmv" -> {
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower, support = support)
            val x0 = Fixtures.vector(d[0], 2)
            val x = x0.copyOf()
            val solve = case.operation == "sptrsv"
            val triangle = SparseReference.dense(a)
            val expected = if (solve) {
                DenseReference.trsv(triangle, x0, lower, transpose, unit)
            } else {
                DenseReference.trmv(triangle, x0, lower, transpose, unit)
            }
            arm(
                verify = {
                    x0.copyInto(x)
                    if (solve) engine.trsv(a, x, lower, transpose, unit) else engine.trmv(a, x, lower, transpose, unit)
                    DenseReference.check(expected, x, "${case.id} triangular")
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
            val triangle = SparseReference.dense(a)
            val expected = if (solve) {
                DenseReference.trsm(triangle, original, lower, transpose, unit, right, alpha)
            } else {
                DenseReference.trmm(triangle, original, lower, transpose, unit, right, alpha)
            }
            arm(
                verify = {
                    original.values.copyInto(b.values)
                    if (solve) {
                        engine.trsm(a, b, lower, transpose, unit, right, alpha, scratch = scratch)
                    } else {
                        engine.trmm(a, b, lower, transpose, unit, right, alpha, scratch = scratch)
                    }
                    DenseReference.check(expected, b.values, "${case.id} triangular block")
                },
            ) {
                original.values.copyInto(b.values)
                if (solve) {
                    engine.trsm(a, b, lower, transpose, unit, right, alpha, scratch = scratch)
                } else {
                    engine.trmm(a, b, lower, transpose, unit, right, alpha, scratch = scratch)
                }
                b.values[0]
            }
        }

        "spsyrk-dense" -> {
            val (n, k) = d
            val a = Fixtures.sparse(n, k, density, 1, support = support)
            val c0 = Fixtures.matrix(n, n, 2)
            val c = Fixtures.matrix(n, n, 2)
            val reference = SparseReference.dense(a)
            arm(
                verify = {
                    c0.values.copyInto(c.values)
                    engine.syrk(alpha, a, false, beta, c, lower, scratch = scratch)
                    DenseReference.check(
                        DenseReference.syrk(alpha, reference, reference, false, beta, c0, lower, doubled = false),
                        c.values, "${case.id} rank update",
                    )
                },
            ) {
                c0.values.copyInto(c.values)
                engine.syrk(alpha, a, false, beta, c, lower, scratch = scratch)
                c.values[0]
            }
        }

        else -> null
    }
}

/** A timed row for a sparse matrix call on [engine], refused when one of the case's extents is empty. */
private fun publish(engine: KoblasEngine, case: BenchCase, timing: String, verify: () -> Unit, run: () -> Double) =
    level23(engine, case, timing, verify, run)?.let { ArmChoice(it, null) } ?: declined()
