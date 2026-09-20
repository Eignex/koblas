package com.eignex.koblas.testutil.allocation

import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory

/** The JVM's per-thread allocation counter, which is where a byte-level measurement has to come from. */
private val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean

/** Holds each block's result so escape analysis cannot delete the allocation being measured. */
internal var allocationSink: Any? = null

/** Bytes allocated by one invocation of [block]. */
internal fun allocatedBytes(block: () -> Any?): Long {
    val id = Thread.currentThread().threadId()
    val before = bean.getThreadAllocatedBytes(id)
    allocationSink = block()
    return bean.getThreadAllocatedBytes(id) - before
}

/**
 * Bytes [block] allocates per call, as the smallest of at least [windows] measurement windows of
 * [iterations] calls, sampling further windows until one comes in at [expected].
 *
 * A vectorised loop allocates one object per vector operation until C2 compiles it and escape analysis
 * removes them. Until that lands the loop allocates steadily, so an uncompiled loop looks exactly as settled
 * as a finished one and waiting for the number to hold still does not work; waiting for it to reach
 * [expected] waits for the event that matters, bounded by [MAX_WINDOWS].
 *
 * [block] deliberately takes no iteration index. A `(Int) -> Any?` would box one on every call, and above
 * `Integer`'s cache that is a sixteen-byte allocation charged to whatever is being measured.
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
 * A named interface rather than a function type, because a `() -> Double` returns its result boxed unless
 * the compiler inlines the call, and several probes through one measurement loop is where it stops doing
 * that; the box was then charged to the kernels as twenty-four bytes a call.
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
