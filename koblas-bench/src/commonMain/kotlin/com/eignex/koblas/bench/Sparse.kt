package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.sparse.SparsePrimitives
import com.eignex.koblas.sparse.SparseRoute

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
        else -> null
    }
}

/** The attribution a sparse row carries, taken from the binding rather than rebuilt from the mode string. */
internal fun sparseKernel(route: SparseRoute): String = "${route.implementation}/${route.entryPoint}"

/** A nonzero scatter epoch; the accumulator is cleared between iterations, so one value serves every pass. */
private const val EPOCH = 1
