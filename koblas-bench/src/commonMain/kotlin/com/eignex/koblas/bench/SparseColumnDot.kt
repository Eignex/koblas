package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.SparseOperation

/** Matched CSC column windows, including public validation and the leaf the selected engine reaches. */
internal fun sparseColumnDotArm(case: BenchCase, engine: KoblasEngine): ArmChoice {
    val rows = case.dimension(0)
    val column = Fixtures.sparseVector(rows, case.option("density", "0.01").toDouble(), 1)
    val count = column.nnz
    val indices = intArrayOf(0, 0, 0) + column.copyIndices() + intArrayOf(0)
    val values = doubleArrayOf(Double.NaN, Double.NaN, Double.NaN) + column.values + doubleArrayOf(Double.NaN)
    val matrix = SparseMatrix.wrap(rows, 5, intArrayOf(0, 1, 2, 3, count + 3, count + 4), indices, values)
    val dense = Fixtures.vector(rows, 2)
    val kernels = engine.sparseKernels
    val raw = case.operation == "spdot-column-raw"
    val run = if (raw) {
        { kernels.dot(indices, 3, values, 3, count, dense) }
    } else {
        { kernels.dotColumn(matrix, 3, dense) }
    }
    val expected = BuiltinEngines.scalar.sparseKernels.dot(indices, 3, values, 3, count, dense)
    DenseReference.check(doubleArrayOf(expected), doubleArrayOf(run()), case.id)
    val route = engine.routeOf(SparseOperation.DotDense, count)
    val validation = if (raw) "portable-index-validation" else "portable-column-validation"
    return ArmChoice(CaseWork(
        "composed", "arithmetic", run, kernel = "$validation+${vectorKernel(route)}",
    ), null)
}
