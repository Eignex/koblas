package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasEngine
import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory

/**
 * Runs allocation checks for the Vector API dense panels in an uninstrumented JVM.
 *
 * Kover's test instrumentation prevents HotSpot from scalar-replacing Vector API carriers, which turns an
 * allocation-free panel into a coverage artifact, so this runs through the `simdDenseAllocationCheck` Gradle
 * task rather than a test task.
 *
 * Whole operations, not only raw panels. A panel measured on its own can be allocation-free while the caller
 * around it is not, and it is the caller a user writes; the raw panels are here beside them so a regression
 * says which of the two moved.
 */
internal object SimdDenseAllocationCheck {
    private const val ORDER = 512
    private const val COLUMNS = 8
    private const val MAX_BYTES_PER_CALL = 8.0
    private const val WARMUP_ITERATIONS = 20_000
    private const val MEASUREMENT_ITERATIONS = 2_000
    private const val MEASUREMENT_WINDOWS = 8

    private val allocationBean = ManagementFactory.getThreadMXBean() as ThreadMXBean

    @Volatile
    private var resultSink = 0.0

    @JvmStatic
    fun main(args: Array<String>) {
        val engine = requireNotNull(BuiltinEngines.simd) {
            "dense allocation check requires the Vector API panel candidate"
        }
        val panels = engine.panelKernels
        require(panels === SimdPanelKernels) { "dense allocation check selected ${panels.name}" }
        require(panels.implementationFor(PanelWork.MultiDot, ORDER, COLUMNS) == panels.name) {
            "dense allocation check uses $ORDER rows, which does not reach the vector body"
        }

        val a = DoubleArray(ORDER * ORDER) { 1.0 + (it % 13) * 0.125 }
        val x = DoubleArray(ORDER) { 1.0 + (it % 7) * 0.25 }
        val y = DoubleArray(ORDER)
        val coefficients = DoubleArray(ORDER) { 0.5 + (it % 5) * 0.125 }
        val sums = DoubleArray(ORDER)

        assertAllocationFree("multi-dot panel") {
            panels.multiDot(0.875, a, 0, ORDER, x, 0, 1, ORDER, COLUMNS, 0.0, y, 0, 1)
            y[0]
        }
        assertAllocationFree("column-update panel") {
            panels.columnUpdate(0.875, a, 0, ORDER, x, 0, 1, ORDER, COLUMNS, y, 0, 1)
            y[0]
        }
        assertAllocationFree("coupled panel") {
            panels.coupledUpdateDot(0.875, a, 0, ORDER, x, 0, ORDER, COLUMNS, y, 0, coefficients, 0, sums, 0)
            sums[0]
        }
        assertAllocationFree("rank-update panel") {
            panels.rankUpdate(0.875, a, 0, ORDER, x, 0, 1, ORDER, COLUMNS, coefficients, 0, 1)
            a[0]
        }
        checkWholeOperations(engine)
    }

    /** The complete callers, where a staging copy or a wrapper would show up that a raw panel cannot. */
    private fun checkWholeOperations(engine: KoblasEngine) {
        val matrix = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 1.0 + (it % 13) * 0.125 })
        val triangle = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 0.25 + (it % 11) * 0.0625 })
        for (i in 0 until ORDER) triangle.values[i + i * ORDER] = 4.0
        val x = DoubleArray(ORDER) { 1.0 + (it % 7) * 0.25 }
        val y = DoubleArray(ORDER) { 0.5 }
        val target = DoubleArray(ORDER) { 1.0 }
        val vector = DenseVector.wrap(x)

        assertAllocationFree("gemv") {
            engine.gemv(0.875, matrix, x, -0.25, y)
            y[0]
        }
        assertAllocationFree("gemv transposed") {
            engine.gemv(0.875, matrix, x, -0.25, y, transpose = true)
            y[0]
        }
        assertAllocationFree("symv") {
            engine.symv(0.875, matrix, x, -0.25, y)
            y[0]
        }
        assertAllocationFree("ger") {
            engine.ger(1e-12, x, y, matrix)
            matrix.values[0]
        }
        assertAllocationFree("syr") {
            engine.syr(1e-12, vector, matrix)
            matrix.values[0]
        }
        assertAllocationFree("trmv") {
            target.fill(1.0)
            engine.trmv(triangle, target, lower = true)
            target[0]
        }
        assertAllocationFree("trsv") {
            target.fill(1.0)
            engine.trsv(triangle, target, lower = true)
            target[0]
        }
    }

    private fun assertAllocationFree(name: String, block: Work) {
        val bytes = bytesPerIteration(block)
        check(bytes <= MAX_BYTES_PER_CALL) { "$name allocated $bytes B per call" }
        println("$name allocated $bytes B per call")
    }

    /**
     * One measured call.
     *
     * A named interface rather than a function type, because a `() -> Double` returns its result boxed and
     * that box is charged to whatever is being measured. This one compiles to a primitive return, so the
     * number is the call's own allocation and nothing else.
     */
    private fun interface Work {
        fun run(): Double
    }

    private fun bytesPerIteration(block: Work): Double {
        repeat(WARMUP_ITERATIONS) { resultSink = block.run() }
        val id = Thread.currentThread().threadId()
        var best = Double.MAX_VALUE
        repeat(MEASUREMENT_WINDOWS) {
            val before = allocationBean.getThreadAllocatedBytes(id)
            repeat(MEASUREMENT_ITERATIONS) { resultSink = block.run() }
            val after = allocationBean.getThreadAllocatedBytes(id)
            best = minOf(best, (after - before).toDouble() / MEASUREMENT_ITERATIONS)
        }
        return best
    }
}
