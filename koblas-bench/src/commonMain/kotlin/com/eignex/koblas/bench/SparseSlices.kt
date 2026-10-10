package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.vendor.RouteKind
import kotlin.math.sqrt

/** Raw CSC windows through the selected engine, including validation and any indexed fallback. */
internal fun sparseSlicesArm(case: BenchCase, engine: KoblasEngine): ArmChoice {
    if (engine !== koblas) return ArmChoice(null, "raw slice cases use the platform-selected engine")
    val size = case.dimension(0)
    val sparse = Fixtures.sparseVector(size, case.option("density", "0.01").toDouble(), 1)
    val count = sparse.nnz
    val indices = IntArray(count + INDEX_OFFSET + 2) { -1 }
    sparse.copyIndices().copyInto(indices, INDEX_OFFSET)
    val values = DoubleArray(count + VALUE_OFFSET + 2) { Double.NaN }
    sparse.values.copyInto(values, VALUE_OFFSET)
    val initial = Fixtures.vector(size, 2)
    val destination = initial.copyOf()
    val scalar = BuiltinEngines.scalar.sparseKernels
    val kernels = engine.sparseKernels
    val operation = when (case.operation) {
        "spdot-slice" -> SparseOperation.DotDense
        "spnrm2-slice" -> SparseOperation.IndexedNrm2
        "spaxpy-slice" -> SparseOperation.Axpy
        else -> SparseOperation.Scatter
    }
    val mutation = operation == SparseOperation.Axpy || operation == SparseOperation.Scatter
    val run = {
        when (operation) {
            SparseOperation.DotDense -> kernels.dot(indices, INDEX_OFFSET, values, VALUE_OFFSET, count, initial)
            SparseOperation.IndexedNrm2 -> kernels.nrm2(indices, INDEX_OFFSET, count, initial)
            else -> {
                initial.copyInto(destination)
                if (operation == SparseOperation.Axpy) {
                    kernels.axpy(destination, 0.875, indices, INDEX_OFFSET, values, VALUE_OFFSET, count)
                } else {
                    kernels.scatter(indices, INDEX_OFFSET, values, VALUE_OFFSET, count, destination)
                }
                destination[indices[INDEX_OFFSET]]
            }
        }
    }
    if (mutation) {
        val expected = initial.copyOf()
        if (operation == SparseOperation.Axpy) {
            scalar.axpy(expected, 0.875, indices, INDEX_OFFSET, values, VALUE_OFFSET, count)
        } else {
            scalar.scatter(indices, INDEX_OFFSET, values, VALUE_OFFSET, count, expected)
        }
        run()
        DenseReference.check(expected, destination, case.id)
    } else {
        var sum = 0.0
        for (k in 0 until count) {
            val selected = initial[indices[INDEX_OFFSET + k]]
            sum += if (operation == SparseOperation.DotDense) values[VALUE_OFFSET + k] * selected else selected * selected
        }
        val expected = if (operation == SparseOperation.DotDense) sum else sqrt(sum)
        DenseReference.check(doubleArrayOf(expected), doubleArrayOf(run()), case.id)
    }
    val route = engine.routeOf(operation, count)
    val kernel = if (route.kind == RouteKind.Composed) {
        "${route.implementation}+scalar-rescale/${route.entryPoint}"
    } else {
        vectorKernel(route)
    }
    return ArmChoice(CaseWork("default-policy", if (mutation) "reset-and-arithmetic" else "arithmetic", run,
        result = if (mutation) destination else null, kernel = "portable-index-validation+$kernel"), null)
}

private const val INDEX_OFFSET = 3
private const val VALUE_OFFSET = 5
