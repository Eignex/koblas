package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.SparseMatrix

internal fun sparseWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    if (!case.operation.startsWith("sp")) return null
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
        "spdot-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.vector(d[0], 2)
            val indices = x.copyIndices()
            CaseWork("unsupported", "arithmetic", {
                engine.sparseKernels.dot(indices, 0, x.values, 0, x.values.size, y)
            })
        }
        "spdot-sparse" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y = Fixtures.sparseVector(d[0], density, 2)
            CaseWork("unsupported", "arithmetic", { engine.sparseKernels.dot(x, y) })
        }
        "spaxpy" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", { y0.copyInto(y); engine.sparseKernels.axpy(y, 0.875, x); y[0] })
        }
        "spaxpy-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            CaseWork("unsupported", "reset-and-arithmetic", {
                y0.copyInto(y)
                engine.sparseKernels.axpy(y, 0.875, indices, 0, x.values, 0, x.values.size)
                y[0]
            })
        }
        "spnrm2", "spasum" -> {
            val x = Fixtures.sparseVector(d[0], density, 1)
            CaseWork("direct", "arithmetic", { if (case.operation == "spnrm2") engine.sparseKernels.nrm2(x) else engine.sparseKernels.asum(x) })
        }
        "spnrm2-indexed" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val dense = Fixtures.vector(d[0], 2)
            CaseWork("unsupported", "arithmetic", {
                engine.sparseKernels.nrm2(indices, 0, indices.size, dense)
            })
        }
        "spscatter-raw" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val indices = x.copyIndices()
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            CaseWork("unsupported", "reset-and-arithmetic", {
                dense0.copyInto(dense)
                engine.sparseKernels.scatter(indices, 0, x.values, 0, x.values.size, dense)
                dense[0]
            })
        }
        "spgather" -> {
            val x = Fixtures.sparseVector(d[0], density, 1)
            val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            val timing = case.option("timing", "reset-and-arithmetic")
            CaseWork("direct", timing, {
                if (timing == "reset-and-arithmetic") dense0.copyInto(dense)
                engine.sparseKernels.gather(x, dense)
                if (x.values.isEmpty()) 0.0 else x.values[0] + x.values[x.values.lastIndex]
            }, result = x.values)
        }
        "spscatter", "spgather-zero" -> {
            val x = Fixtures.sparseVector(d[0], density, 1); val values0 = x.values.copyOf(); val dense0 = Fixtures.vector(d[0], 2); val dense = dense0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", {
                values0.copyInto(x.values); dense0.copyInto(dense)
                when (case.operation) {
                    "spscatter" -> engine.sparseKernels.scatter(x, dense)
                    else -> engine.sparseKernels.gatherZero(x, dense)
                }
                x.values.firstOrNull() ?: dense[0]
            })
        }
        else -> null
    }
}


