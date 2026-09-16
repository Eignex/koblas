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
 * removes them, and that compilation is triggered by invocation count and finished on another thread. Until
 * it lands the loop allocates steadily, so waiting for the measurement to hold still does not work: an
 * uncompiled loop looks exactly as settled as a finished one, and on a small runner the difference decides
 * the result run by run. Waiting for the number to reach [expected] instead waits for the event that
 * actually matters. [MAX_WINDOWS] bounds that wait, so a loop that really does allocate is reported rather
 * than waited on forever, and the run says it gave up.
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
    repeat(warmup) { allocationSink = block() } // let the JIT settle, since the first calls allocate profiling data
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
