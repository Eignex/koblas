package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.sparse.SparseRoute

/**
 * The sparse work for one case on [engine], or the reason this engine cannot measure it, or null when the case
 * is not a sparse one.
 *
 * Every arm here is an exact comparison: the row is published under the engine that was selected, so a call the
 * engine hands to a different implementation is declined rather than timed. A Vector API selection does that
 * for a support below its crossover, for an operation it never vectorised, and on a host whose indexed loads or
 * stores it cannot use, and the route is what distinguishes those from a call it really ran.
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
        else -> null
    }
}

/** The attribution a sparse row carries, taken from the binding rather than rebuilt from the mode string. */
internal fun sparseKernel(route: SparseRoute): String = "${route.implementation}/${route.entryPoint}"
