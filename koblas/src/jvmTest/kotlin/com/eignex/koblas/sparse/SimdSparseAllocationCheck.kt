package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory

/**
 * Runs allocation checks for the Vector API indexed sparse paths in an uninstrumented JVM.
 *
 * Kover's test instrumentation prevents HotSpot from scalar-replacing Vector API carriers, so this intentionally
 * runs through the `simdSparseAllocationCheck` Gradle task instead of a test task.
 */
internal object SimdSparseAllocationCheck {
    internal const val ENTRY_COUNT = 512
    private const val DIMENSION = ENTRY_COUNT * 4
    private const val MAX_BYTES_PER_CALL = 64.0
    private const val WARMUP_ITERATIONS = 20_000
    private const val MEASUREMENT_ITERATIONS = 10_000
    private const val MEASUREMENT_WINDOWS = 3

    private val allocationBean = ManagementFactory.getThreadMXBean() as ThreadMXBean

    /** A volatile result keeps read-only dot and norm calls observable to the optimizer. */
    @Volatile
    private var resultSink = 0.0

    @JvmStatic
    fun main(args: Array<String>) {
        val engine = requireNotNull(BuiltinEngines.simd) {
            "SIMD sparse allocation check requires the Vector API module"
        }
        require(engine.sparseKernels.name == "simd-sparse") {
            "SIMD sparse allocation check selected ${engine.sparseKernels.name}"
        }
        require(SparseSimd.autoIndexedLoadEligible) {
            "SIMD sparse allocation check requires a preferred species with more than one lane"
        }
        require(crossesSimdCrossover(SparseTuning.simdIndexedCrossover)) {
            "SIMD sparse allocation check uses $ENTRY_COUNT entries below " +
                "the ${SparseTuning.simdIndexedCrossover}-entry crossover"
        }

        val indices = IntArray(ENTRY_COUNT) { 1 + it * 4 }
        val values = DoubleArray(ENTRY_COUNT) { it * 0.125 - 16.0 }
        val dense = DoubleArray(DIMENSION) { 1.0 + (it % 17) * 0.03125 }
        val kernels = engine.sparseKernels

        assertAllocationFree("indexed dot") {
            kernels.dot(indices, 0, values, 0, ENTRY_COUNT, dense)
        }
        assertAllocationFree("indexed gather") {
            // Exercise the Vector API leaf even where production prefers scalar indexed loads.
            SparseSimd.gather(indices, 0, values, 0, ENTRY_COUNT, dense)
            values[ENTRY_COUNT / 2]
        }
        assertAllocationFree("indexed norm") {
            kernels.nrm2(indices, 0, ENTRY_COUNT, dense)
        }
    }

    internal fun crossesSimdCrossover(crossover: Int): Boolean = ENTRY_COUNT >= crossover

    private fun assertAllocationFree(name: String, block: () -> Double) {
        val bytes = bytesPerIteration(block)
        check(bytes <= MAX_BYTES_PER_CALL) { "$name allocated $bytes B per call" }
    }

    private fun bytesPerIteration(block: () -> Double): Double {
        repeat(WARMUP_ITERATIONS) { resultSink = block() }
        val id = Thread.currentThread().threadId()
        var best = Double.MAX_VALUE
        repeat(MEASUREMENT_WINDOWS) {
            val before = allocationBean.getThreadAllocatedBytes(id)
            repeat(MEASUREMENT_ITERATIONS) { resultSink = block() }
            val after = allocationBean.getThreadAllocatedBytes(id)
            best = minOf(best, (after - before).toDouble() / MEASUREMENT_ITERATIONS)
        }
        return best
    }
}
