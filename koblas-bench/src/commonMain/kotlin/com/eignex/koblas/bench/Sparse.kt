package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.PreparedSparseMatrix
import com.eignex.koblas.sparse.SparseWorkspace

internal fun sparseWork(case: BenchCase, engine: KoblasContext): CaseWork? {
    if (!case.operation.startsWith("sp") && !case.operation.startsWith("workspace-")) return null
    val density = case.option("density", "0.01").toDouble()
    val d = case.dimensions
    val lower = case.option("uplo", "L") == "L"
    val transpose = case.flag("transA")
    val unit = case.option("diag", "N") == "U"
    val mode = case.option("mode", "oneshot")
    return when (case.operation) {
        "spdot" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.vector(d[0], 2)
            CaseWork("direct", "arithmetic", { engine.sparseKernels.dot(x, y) })
        }
        "spdot-sparse" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.sparseVector(d[0], density, 2)
            CaseWork("unsupported", "arithmetic", { engine.sparseKernels.dot(x, y) })
        }
        "spaxpy" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", { y0.copyInto(y); engine.sparseKernels.axpy(y, 0.875, x); y[0] })
        }
        "spnrm2", "spasum" -> {
            val x = Fixtures.sparseVector(d[0], density, 1)
            CaseWork("unsupported", "arithmetic", { if (case.operation == "spnrm2") engine.sparseKernels.nrm2(x) else engine.sparseKernels.asum(x) })
        }
        "spscatter", "spgather", "spgather-zero" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val values0 = x.values.copyOf(); val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", {
                values0.copyInto(x.values); dense0.copyInto(dense)
                when (case.operation) {
                    "spscatter" -> engine.sparseKernels.scatter(x, dense)
                    "spgather" -> engine.sparseKernels.gather(x, dense)
                    else -> engine.sparseKernels.gatherZero(x, dense)
                }
                x.values.firstOrNull() ?: dense[0]
            })
        }
        "spgemv" -> {
            val a = Fixtures.sparse(d[0], d[1], density, 1); val x = Fixtures.vector(d[1], 2)
            val y0 = Fixtures.vector(d[0], 3); val y = y0.copyOf()
            preparedOrOneShot(mode, engine, a,
                preparedRun = { prepared -> y0.copyInto(y); prepared.gemv(0.875, x, -0.25, y); y[0] },
                oneShotRun = { y0.copyInto(y); engine.gemv(0.875, a, x, -0.25, y); y[0] },
            )
        }
        "spmm" -> {
            val (m, n, k) = d; val a = Fixtures.sparse(m, k, density, 1); val b = Fixtures.matrix(k, n, 2)
            val c0 = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3)
            preparedOrOneShot(mode, engine, a,
                preparedRun = { prepared -> c0.data.copyInto(c.data); prepared.gemm(0.875, false, b, -0.25, c); c.data[0] },
                oneShotRun = { c0.data.copyInto(c.data); engine.gemm(0.875, a, false, b, false, -0.25, c); c.data[0] },
            )
        }
        "spgemm" -> {
            val (m, n, k) = d; val a = Fixtures.sparse(m, k, density, 1); val b = Fixtures.sparse(k, n, density, 2)
            preparedOrOneShot(mode, engine, a, comparison = "partial",
                preparedRun = { prepared -> prepared.gemm(b).values.firstOrNull() ?: 0.0 },
                oneShotRun = { engine.gemm(a, b).values.firstOrNull() ?: 0.0 },
            )
        }
        "spsymv" -> {
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower)
            val x = Fixtures.vector(d[0], 2); val y0 = Fixtures.vector(d[0], 3); val y = y0.copyOf()
            CaseWork("direct", "oneshot", { y0.copyInto(y); engine.symv(0.875, a, x, -0.25, y, lower); y[0] })
        }
        "spsymm" -> {
            val (n, rhs) = d; val a = Fixtures.sparse(n, n, density, 1, triangular = true, lower = lower)
            val b = Fixtures.matrix(n, rhs, 2); val c0 = Fixtures.matrix(n, rhs, 3); val c = Fixtures.matrix(n, rhs, 3)
            CaseWork("direct", "oneshot", { c0.data.copyInto(c.data); engine.symm(0.875, a, b, -0.25, c, lower); c.data[0] })
        }
        "sptrsv", "sptrmv" -> {
            val a = Fixtures.sparse(d[0], d[0], density, 1, triangular = true, lower = lower)
            val x0 = Fixtures.vector(d[0], 2); val x = x0.copyOf()
            CaseWork("direct", "oneshot", {
                x0.copyInto(x)
                if (case.operation == "sptrsv") engine.trsv(a, x, lower, transpose, unit) else engine.trmv(a, x, lower, transpose, unit)
                x[0]
            })
        }
        "sptrsm", "sptrmm" -> {
            val (order, rhs) = d; val right = case.option("side", "L") == "R"
            val a = Fixtures.sparse(order, order, density, 1, triangular = true, lower = lower)
            val original = Fixtures.matrix(if (right) rhs else order, if (right) order else rhs, 2)
            val b = Fixtures.matrix(original.rows, original.cols, 2)
            CaseWork("direct", "oneshot", {
                original.data.copyInto(b.data)
                if (case.operation == "sptrsm") engine.trsm(a, b, lower, transpose, unit, right, 0.875)
                else engine.trmm(a, b, lower, transpose, unit, right, 0.875)
                b.data[0]
            })
        }
        "spsyrk-dense", "spsyrk-sparse" -> {
            val (n, k) = d; val a = Fixtures.sparse(n, k, density, 1); val c0 = Fixtures.matrix(n, n, 2); val c = Fixtures.matrix(n, n, 2)
            CaseWork(if (case.operation == "spsyrk-sparse") "partial" else "direct", "oneshot", {
                if (case.operation == "spsyrk-dense") {
                    c0.data.copyInto(c.data); engine.syrk(0.875, a, false, -0.25, c, lower); c.data[0]
                } else engine.syrk(a, false, lower).values.firstOrNull() ?: 0.0
            })
        }
        "spadd" -> {
            val (m, n) = d; val a = Fixtures.sparse(m, n, density, 1); val b = Fixtures.sparse(m, n, density, 2)
            CaseWork("partial", "oneshot", { engine.addScaled(0.875, a, false, b).values.firstOrNull() ?: 0.0 })
        }
        else -> workspaceWork(case, density)
    }
}

private fun preparedOrOneShot(
    mode: String,
    engine: KoblasContext,
    matrix: SparseMatrix,
    comparison: String = "direct",
    preparedRun: (PreparedSparseMatrix) -> Double,
    oneShotRun: () -> Double,
): CaseWork {
    if (mode == "oneshot") {
        return CaseWork(comparison, "oneshot", oneShotRun)
    }
    val prepared = engine.prepare(matrix)
    return CaseWork(comparison, "prepared", { preparedRun(prepared) }, { prepared.close() })
}

private fun workspaceWork(case: BenchCase, density: Double): CaseWork {
    val dimension = case.dimension(0)
    val count = (dimension * density + 0.5).toInt().coerceAtLeast(1)
    val sparse = Fixtures.sparseVector(dimension, density, 1)
    val indices = sparse.copyIndices(); val values = sparse.values
    val accumulator = DoubleArray(dimension); val marks = IntArray(dimension); val touched = IntArray(count)
    val outIndices = IntArray(count); val outValues = DoubleArray(count); val active = BooleanArray(dimension) { it % 3 != 0 }
    var epoch = 1
    SparseWorkspace.scatterAxpy(1.0, indices, 0, values, 0, count, accumulator, marks, epoch, touched, 0, 0)
    val populated = accumulator.copyOf(); val populatedMarks = marks.copyOf()
    val status = IntArray(1)
    val maximum = SparseWorkspace.activeColumnMaxAbs(indices, 0, values, 0, count, active)
    return CaseWork("unsupported", "workspace", {
        when (case.operation) {
            "workspace-scatter" -> {
                epoch++; accumulator.fill(0.0); marks.fill(0)
                SparseWorkspace.scatterAxpy(0.875, indices, 0, values, 0, count, accumulator, marks, epoch, touched, 0, 0).toDouble()
            }
            "workspace-scatter-checked" -> {
                epoch++; accumulator.fill(0.0); marks.fill(0); status[0] = 0
                SparseWorkspace.scatterAxpyChecked(0.875, indices, 0, values, 0, count, accumulator, marks, epoch, touched, 0, 0, status, 0).toDouble()
            }
            "workspace-gather" -> SparseWorkspace.gatherTouched(touched, 0, count, populated, outIndices, 0, outValues, 0).toDouble()
            "workspace-gather-clear" -> {
                populated.copyInto(accumulator); populatedMarks.copyInto(marks)
                SparseWorkspace.gatherClearTouched(touched, 0, count, accumulator, marks, outIndices, 0, outValues, 0).toDouble()
            }
            "workspace-max" -> SparseWorkspace.activeColumnMaxAbs(indices, 0, values, 0, count, active)
            else -> SparseWorkspace.pivotCandidatePositions(indices, 0, values, 0, count, active, maximum, 0.01, 0.1, outIndices, 0).toDouble()
        }
    })
}
