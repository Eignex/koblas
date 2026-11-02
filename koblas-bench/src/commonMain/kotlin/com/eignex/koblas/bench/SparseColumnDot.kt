package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.SparseMatrix

/** Matched CSC column windows, including public validation, through the selected engine. */
internal fun sparseColumnDotArm(case: BenchCase, engine: KoblasEngine): ArmChoice {
    val rows = case.dimension(0)
    val column = Fixtures.sparseVector(rows, case.option("density", "0.01").toDouble(), 1)
    val count = column.nnz
    val indices = intArrayOf(0, 0, 0) + column.copyIndices() + intArrayOf(0)
    val values = doubleArrayOf(Double.NaN, Double.NaN, Double.NaN) + column.values + doubleArrayOf(Double.NaN)
    val matrix = SparseMatrix.wrap(rows, 5, intArrayOf(0, 1, 2, 3, count + 3, count + 4), indices, values)
    val dense = Fixtures.vector(rows, 2)
    val kernels = engine
    val raw = case.operation == "spdot-column-raw"
    val run = if (raw) {
        { kernels.dot(indices, 3, values, 3, count, dense) }
    } else {
        { kernels.dotColumn(matrix, 3, dense) }
    }
    val expected = BuiltinEngines.scalar.dot(indices, 3, values, 3, count, dense)
    DenseReference.check(doubleArrayOf(expected), doubleArrayOf(run()), case.id)
    return ArmChoice(engineWork(engine, case.operation, "arithmetic", run), null)
}
