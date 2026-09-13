package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.sparse.SparseSlices

internal val sparseSlicesComparisonOperations = setOf(
    "sparse-slices-cycle", "sparse-slices-cycle-checked", "sparse-slices-gather",
    "sparse-slices-gather-clear", "sparse-slices-clear", "sparse-slices-reduce-dot-checked",
    "sparse-slices-reduce-dot-unchecked",
)

internal fun sparseSlicesTiming(operation: String): String = when (operation) {
    "sparse-slices-cycle" -> "slices-cycle-v1"
    "sparse-slices-cycle-checked" -> "slices-cycle-checked-v1"
    "sparse-slices-gather" -> "slices-gather-v1"
    "sparse-slices-gather-clear" -> "slices-refill-gather-clear-v1"
    "sparse-slices-clear" -> "slices-refill-clear-v1"
    "sparse-slices-reduce-dot-checked" -> "slices-ordered-dot-checked-v1"
    "sparse-slices-reduce-dot-unchecked" -> "slices-dot-v1"
    else -> error("not a sparse slice comparison: $operation")
}

/** All buffers survive timed calls; the cycle itself returns touched scratch to its reusable state. */
internal class SparseSlicesReuseWork(private val case: BenchCase, private val engine: KoblasEngine) {
    private val sparse = Fixtures.sparseVector(case.dimension(0), case.option("density", "0.01").toDouble(), 1)
    val indices = sparse.copyIndices()
    val values = sparse.values.copyOf()
    val count = indices.size
    val accumulator = DoubleArray(case.dimension(0))
    val marks = IntArray(accumulator.size)
    val touched = IntArray(count)
    val outIndices = IntArray(count)
    val outValues = DoubleArray(count)
    val status = IntArray(1)
    var written = 0
        private set
    private val compact = case.option("compact", "N") == "T"

    init {
        if (case.option("locality", "sorted") == "shuffled") {
            val random = Fixtures.stream(71)
            for (i in count - 1 downTo 1) {
                val j = (random.nextLong().ushr(1) % (i + 1)).toInt()
                val index = indices[i]; indices[i] = indices[j]; indices[j] = index
                val value = values[i]; values[i] = values[j]; values[j] = value
            }
        }
        indices.copyInto(touched)
        if (!case.operation.startsWith("sparse-slices-cycle")) refill()
    }

    private fun refill() {
        for (k in 0 until count) {
            accumulator[touched[k]] = values[k]
            marks[touched[k]] = 1
        }
    }

    fun run(): Double {
        when (case.operation) {
            "sparse-slices-cycle", "sparse-slices-cycle-checked" -> {
                status[0] = 0
                val first = (count + 1) / 2
                val checked = case.operation == "sparse-slices-cycle-checked"
                val touchedCount = if (checked) {
                    SparseSlices.scatterAxpyChecked(0.875, indices, 0, values, 0, first,
                        accumulator, marks, 1, touched, 0, 0, status, 0)
                } else {
                    SparseSlices.scatterAxpy(0.875, indices, 0, values, 0, first,
                        accumulator, marks, 1, touched, 0, 0)
                }
                val total = if (checked) {
                    SparseSlices.scatterAxpyChecked(-0.875, indices, 0, values, 0, count,
                        accumulator, marks, 1, touched, 0, touchedCount, status, 0)
                } else {
                    SparseSlices.scatterAxpy(-0.875, indices, 0, values, 0, count,
                        accumulator, marks, 1, touched, 0, touchedCount)
                }
                written = SparseSlices.gatherClearTouched(touched, 0, total, accumulator, marks,
                    outIndices, 0, outValues, 0, compact)
            }
            "sparse-slices-gather" -> written = SparseSlices.gatherTouched(touched, 0, count,
                accumulator, outIndices, 0, outValues, 0, compact)
            "sparse-slices-gather-clear" -> {
                refill()
                written = SparseSlices.gatherClearTouched(touched, 0, count, accumulator, marks,
                    outIndices, 0, outValues, 0, compact)
            }
            "sparse-slices-clear" -> {
                refill()
                SparseSlices.clearTouched(touched, 0, count, accumulator, marks)
                return accumulator[touched[0]] + accumulator[touched[count - 1]] + marks[touched[0]] + marks[touched[count - 1]]
            }
            "sparse-slices-reduce-dot-checked" -> {
                status[0] = 0
                val result = SparseSlices.reduceDotChecked(0.0, false, indices, 0, values, 0, count,
                    accumulator, status, 0)
                return result + status[0]
            }
            "sparse-slices-reduce-dot-unchecked" -> return engine.sparseKernels.dot(indices, 0, values, 0, count, accumulator)
            else -> error("not a sparse slice comparison: ${case.operation}")
        }
        return written.toDouble() + status[0] + if (written == 0) 0.0 else
            outValues[0] + outValues[written - 1] + outIndices[0] + outIndices[written - 1]
    }
}
