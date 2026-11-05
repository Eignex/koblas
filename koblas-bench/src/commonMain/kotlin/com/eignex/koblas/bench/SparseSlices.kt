package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine

/** Raw CSC windows through the selected engine, including validation and any indexed fallback. */
internal fun sparseSlicesArm(case: BenchCase, engine: KoblasEngine): ArmChoice {
    val size = case.dimension(0)
    val sparse = Fixtures.sparseVector(size, case.option("density", "0.01").toDouble(), 1)
    val count = sparse.nnz
    val indices = IntArray(count + INDEX_OFFSET + 2) { -1 }
    sparse.copyIndices().copyInto(indices, INDEX_OFFSET)
    val values = DoubleArray(count + VALUE_OFFSET + 2) { Double.NaN }
    sparse.values.copyInto(values, VALUE_OFFSET)
    val initial = Fixtures.vector(size, 2)
    val destination = initial.copyOf()
    val scalar = KoblasEngine.scalar
    val kernels = engine
    val operation = case.operation
    val mutation = operation == "spaxpy-slice" || operation == "spscatter-slice"
    val run = {
        when (operation) {
            "spdot-slice" -> kernels.dot(indices, INDEX_OFFSET, values, VALUE_OFFSET, count, initial)
            else -> {
                initial.copyInto(destination)
                if (operation == "spaxpy-slice") {
                    kernels.axpy(indices, INDEX_OFFSET, values, VALUE_OFFSET, count, 0.875, destination)
                } else {
                    kernels.scatter(indices, INDEX_OFFSET, values, VALUE_OFFSET, count, destination)
                }
                destination[indices[INDEX_OFFSET]]
            }
        }
    }
    if (mutation) {
        val expected = initial.copyOf()
        if (operation == "spaxpy-slice") {
            scalar.axpy(indices, INDEX_OFFSET, values, VALUE_OFFSET, count, 0.875, expected)
        } else {
            scalar.scatter(indices, INDEX_OFFSET, values, VALUE_OFFSET, count, expected)
        }
        run()
        DenseReference.check(expected, destination, case.id)
    } else {
        var sum = 0.0
        for (k in 0 until count) {
            val selected = initial[indices[INDEX_OFFSET + k]]
            sum += values[VALUE_OFFSET + k] * selected
        }
        val expected = sum
        DenseReference.check(doubleArrayOf(expected), doubleArrayOf(run()), case.id)
    }
    val timing = if (mutation) "reset-and-arithmetic" else "arithmetic"
    return ArmChoice(engineWork(engine, operation, timing, run, if (mutation) destination else null), null)
}

private const val INDEX_OFFSET = 3
private const val VALUE_OFFSET = 5
