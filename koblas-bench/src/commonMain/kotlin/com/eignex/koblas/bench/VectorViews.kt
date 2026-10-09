package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.StridedVector
import com.eignex.koblas.axpy
import com.eignex.koblas.copy
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dot
import com.eignex.koblas.koblas
import com.eignex.koblas.swap

/**
 * Vector entry points include their alias checks and view adaptation. They select the platform engine
 * themselves, so the row names the whole portable wrapper and is measured only on the default-policy arm.
 */
internal fun vectorViewsArm(case: BenchCase, engine: KoblasEngine): ArmChoice {
    if (engine !== koblas) {
        return ArmChoice(null, "vector entry points use the platform-selected engine")
    }
    val size = case.dimension(0)
    if (case.operation == "spdot-view") {
        val sparse = Fixtures.sparseVector(size, case.option("density", "0.01").toDouble(), 1)
        val dense = Fixtures.vector(size, 2)
        val backing = DoubleArray(2 * size) { if (it % 2 == 0) dense[it / 2] else Double.NaN }
        val view = StridedVector(backing, 0, size, 2)
        val expected = BuiltinEngines.scalar.sparseKernels.dot(sparse, dense)
        DenseReference.check(doubleArrayOf(expected), doubleArrayOf(sparse dot view), case.id)
        return ArmChoice(CaseWork("default-policy", "arithmetic", { sparse dot view },
            kernel = "portable-vector+scalar/dot-view"), null)
    }
    val overlap = case.operation == "copy-overlap"
    val initial = Fixtures.vector(if (overlap) size + 1 else size * 2, 1)
    val backing = initial.copyOf()
    val stride = if (overlap) 1 else 2
    val source = StridedVector(backing, 0, size, stride)
    val destination = StridedVector(backing, 1, size, stride)
    val x = DoubleArray(size) { initial[it * stride] }
    val y = DoubleArray(size) { initial[1 + it * stride] }
    val kernels = BuiltinEngines.scalar.vectorKernels
    when (case.operation) {
        "copy-overlap" -> x.copyInto(y)
        "axpy-views" -> kernels.axpy(y, 0, 0.875, x, 0, size)
        "swap-views" -> kernels.swap(x, 0, y, 0, size)
    }
    val expected = initial.copyOf()
    for (i in 0 until size) {
        if (!overlap) expected[i * stride] = x[i]
        expected[1 + i * stride] = y[i]
    }
    val run = {
        initial.copyInto(backing)
        when (case.operation) {
            "copy-overlap" -> copy(source, destination)
            "axpy-views" -> destination.axpy(0.875, source)
            "swap-views" -> swap(source, destination)
        }
        backing[0] + backing.last()
    }
    run()
    DenseReference.check(expected, backing, case.id)
    val kernel = when (case.operation) {
        "copy-overlap" -> "portable-vector+array-copy/copy-overlap"
        else -> {
            val operation = if (case.operation == "axpy-views") DenseOperation.Axpy else DenseOperation.Swap
            "portable-vector+${vectorKernel(engine.routeOf(operation, size, contiguous = false))}/${case.operation}"
        }
    }
    return ArmChoice(CaseWork("default-policy", "reset-and-arithmetic", run, result = backing,
        kernel = kernel), null)
}
