package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.SparseSlices

internal fun sparseWork(case: BenchCase, engine: KoblasEngine): CaseWork? {
    if (!case.operation.startsWith("sp") && !case.operation.startsWith("sparse-slices-")) return null
    if (case.option("timing", "") == "reuse") {
        val work = SparseSlicesReuseWork(case, engine)
        return CaseWork("composed", sparseSlicesTiming(case.operation), { work.run() }, result = work.outValues)
    }
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
        else -> sparseSlicesWork(case, density, engine)
    }
}


private fun sparseSlicesWork(case: BenchCase, density: Double, engine: KoblasEngine): CaseWork {
    val dimension = case.dimension(0)
    val count = (dimension * density + 0.5).toInt().coerceAtLeast(1)
    val sparse = Fixtures.sparseVector(dimension, density, 1)
    val indices = sparse.copyIndices(); val values = sparse.values
    val accumulator = DoubleArray(dimension); val marks = IntArray(dimension); val touched = IntArray(count)
    val outIndices = IntArray(count); val outValues = DoubleArray(count); val active = BooleanArray(dimension) { it % 3 != 0 }
    var epoch = 1
    SparseSlices.scatterAxpy(1.0, indices, 0, values, 0, count, accumulator, marks, epoch, touched, 0, 0)
    val populated = accumulator.copyOf(); val populatedMarks = marks.copyOf()
    val status = IntArray(1)
    val maximum = SparseSlices.activeColumnMaxAbs(indices, 0, values, 0, count, active)
    val clearValues = populated.copyOf(); val clearMarks = populatedMarks.copyOf()
    return CaseWork("unsupported", "sparse-slices", {
        when (case.operation) {
            "sparse-slices-scatter" -> {
                epoch++; accumulator.fill(0.0); marks.fill(0)
                SparseSlices.scatterAxpy(0.875, indices, 0, values, 0, count, accumulator, marks, epoch, touched, 0, 0).toDouble()
            }
            "sparse-slices-scatter-checked" -> {
                epoch++; accumulator.fill(0.0); marks.fill(0); status[0] = 0
                SparseSlices.scatterAxpyChecked(0.875, indices, 0, values, 0, count, accumulator, marks, epoch, touched, 0, 0, status, 0).toDouble()
            }
            "sparse-slices-gather" -> SparseSlices.gatherTouched(touched, 0, count, populated, outIndices, 0, outValues, 0).toDouble()
            "sparse-slices-gather-clear" -> {
                populated.copyInto(accumulator); populatedMarks.copyInto(marks)
                SparseSlices.gatherClearTouched(touched, 0, count, accumulator, marks, outIndices, 0, outValues, 0).toDouble()
            }
            "sparse-slices-clear" -> {
                populated.copyInto(clearValues); populatedMarks.copyInto(clearMarks)
                SparseSlices.clearTouched(touched, 0, count, clearValues, clearMarks)
                clearValues[0]
            }
            "sparse-slices-clear-local" -> {
                populated.copyInto(clearValues); populatedMarks.copyInto(clearMarks)
                for (k in 0 until count) {
                    clearValues[touched[k]] = 0.0
                    clearMarks[touched[k]] = 0
                }
                clearValues[0]
            }
            "sparse-slices-reduce-dot-checked" -> {
                status[0] = 0
                SparseSlices.reduceDotChecked(0.0, false, indices, 0, values, 0, count, populated, status, 0)
            }
            "sparse-slices-reduce-dot-local" -> {
                status[0] = 0
                checkedDotLoop(indices, values, count, populated, status)
            }
            "sparse-slices-reduce-dot-unchecked" ->
                engine.sparseKernels.dot(indices, 0, values, 0, count, populated)
            "sparse-slices-max" -> SparseSlices.activeColumnMaxAbs(indices, 0, values, 0, count, active)
            else -> SparseSlices.pivotCandidatePositions(indices, 0, values, 0, count, active, maximum, 0.01, 0.1, outIndices, 0).toDouble()
        }
    })
}

private fun checkedDotLoop(
    indices: IntArray,
    values: DoubleArray,
    count: Int,
    dense: DoubleArray,
    status: IntArray,
): Double {
    var result = 0.0
    var bits = status[0]
    for (k in 0 until count) {
        val left = values[k]
        val right = dense[indices[k]]
        val product = left * right
        val updated = result + product
        if (!left.isFinite() || !right.isFinite() || !product.isFinite() || !updated.isFinite()) bits = bits or 1
        if (left.isFinite() && right.isFinite() && left != 0.0 && right != 0.0 && product == 0.0) bits = bits or 2
        result = updated
    }
    status[0] = bits
    return result
}
