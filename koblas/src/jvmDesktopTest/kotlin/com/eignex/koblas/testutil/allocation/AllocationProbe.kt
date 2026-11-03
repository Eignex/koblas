package com.eignex.koblas.testutil.allocation

import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory
import kotlin.test.assertTrue

/** The JVM's per-thread allocation counter, which is where a byte-level measurement has to come from. */
private val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean

/** Holds each block's result so escape analysis cannot delete the allocation being measured. */
private var allocationSink: Any? = null

/** Allowance for effects that are not koblas's (instrumentation, index boxing, JIT noise). */
internal const val FLOOR_BYTES = 64.0

/** An array object's own bytes, beside the elements it holds. */
internal const val ARRAY_HEADER_BYTES = 16

/**
 * One case of a parameterized allocation test: [probe] measured over windows of [iterations] calls against
 * [budget] bytes per call. Each case builds its own fixture, so mutations and workspace loans cannot leak
 * between cases.
 */
internal class AllocationCase(
    private val name: String,
    private val iterations: Int,
    private val budget: Double = FLOOR_BYTES,
    private val probe: () -> Any?,
) {
    fun assertWithinBudget() {
        val bytes = bytesPerIteration(iterations, budget, block = probe)

        assertTrue(bytes <= budget, "$name allocated $bytes B per call against $budget")
    }

    override fun toString(): String = name
}

/** Wraps each case as the one-element parameter row JUnit's parameterized runner hands to a constructor. */
internal fun parameterRows(cases: List<AllocationCase>): List<Array<Any>> = cases.map { arrayOf<Any>(it) }

/**
 * Bytes [block] allocates per call, as the smallest of at least [windows] measurement windows of
 * [iterations] calls, sampling further windows until one comes in at [expected].
 *
 * Vector operations allocate until C2 escape analysis removes their carriers. Stable allocation
 * is therefore insufficient evidence of compilation; sampling waits for [expected], bounded by
 * [MAX_WINDOWS].
 *
 * [block] takes no iteration index because boxing indices beyond `Integer`'s cache would be
 * charged to the operation being measured.
 */
internal fun bytesPerIteration(
    iterations: Int,
    expected: Double,
    warmup: Int = 200,
    windows: Int = 5,
    block: () -> Any?,
): Double {
    repeat(warmup) { allocationSink = block() } // let the JIT settle, since the first calls allocate profiling values
    val id = Thread.currentThread().threadId()
    var best = Double.MAX_VALUE
    var taken = 0
    while (taken < windows || (best > expected && taken < MAX_WINDOWS)) {
        val before = bean.getThreadAllocatedBytes(id)
        repeat(iterations) { allocationSink = block() }
        val after = bean.getThreadAllocatedBytes(id)
        best = minOf(best, (after - before).toDouble() / iterations)
        taken++
    }
    if (best > expected) println("allocation probe gave up after $taken windows at $best B per call")
    return best
}

/** The longest this waits for a loop to be compiled, in windows. */
private const val MAX_WINDOWS = 400

/**
 * One measured call of a probe run outside a test task.
 *
 * A named interface avoids boxing the result of a `() -> Double` when the compiler cannot inline
 * the call through the measurement loop.
 */
internal fun interface AllocationProbe {
    fun run(): Double
}

/** Holds each probe's result so escape analysis cannot delete the allocation being measured. */
@Volatile
private var probeSink = 0.0

/**
 * Bytes [block] allocates per call, as the smallest of [windows] measurement windows of [iterations] calls
 * taken after [warmup] calls have let the JIT settle.
 *
 * The smallest window rather than the mean, because a window that caught a compilation or a safepoint
 * measures that event and not the loop.
 */
internal fun bytesPerCall(block: AllocationProbe, warmup: Int, iterations: Int, windows: Int): Double {
    repeat(warmup) { probeSink = block.run() }
    val id = Thread.currentThread().threadId()
    var best = Double.MAX_VALUE
    repeat(windows) {
        val before = bean.getThreadAllocatedBytes(id)
        repeat(iterations) { probeSink = block.run() }
        val after = bean.getThreadAllocatedBytes(id)
        best = minOf(best, (after - before).toDouble() / iterations)
    }
    return best
}
