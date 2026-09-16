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
 * Bytes [block] allocates per call, as the smallest measurement window of [iterations] calls.
 * The minimum is what the loop costs once the JIT has settled; the other windows carry runtime noise.
 *
 * Windows are taken until [windows] of them in a row fail to improve on the best seen, so the wait for the
 * JIT is as long as the machine needs rather than a fixed count. A vectorised loop allocates one object per
 * vector operation until C2 compiles it and escape analysis removes them, and a few thousand calls of a
 * microsecond-long kernel can finish while that compilation is still queued: on a fast host the first window
 * is already clean and this returns in the same time as a fixed count, on a slow or loaded one it keeps
 * measuring until the allocation stops falling. [MAX_WINDOWS] bounds that wait, so a loop the JIT never
 * settles is reported rather than waited on forever.
 *
 * [block] deliberately takes no iteration index. A `(Int) -> Any?` would box one on every call, and above
 * `Integer`'s cache that is a sixteen-byte allocation charged to whatever is being measured.
 */
internal fun bytesPerIteration(iterations: Int, warmup: Int = 200, windows: Int = 5, block: () -> Any?): Double {
    repeat(warmup) { allocationSink = block() } // let the JIT settle, since the first calls allocate profiling data
    val id = Thread.currentThread().threadId()
    var best = Double.MAX_VALUE
    var first = Double.NaN
    var last = Double.NaN
    var stable = 0
    var taken = 0
    while (stable < windows && taken < MAX_WINDOWS) {
        val before = bean.getThreadAllocatedBytes(id)
        repeat(iterations) { allocationSink = block() }
        val after = bean.getThreadAllocatedBytes(id)
        val per = (after - before).toDouble() / iterations
        if (taken == 0) first = per
        last = per
        // A window has to beat the best by a clear margin to count as the JIT still settling; matching it
        // within noise is what a settled loop does.
        stable = if (per < best * IMPROVEMENT) 0 else stable + 1
        best = minOf(best, per)
        taken++
    }
    // Hitting the cap means the loop was still moving when the budget ran out, so the number returned is a
    // floor on what it costs rather than what it settles at. Say so: the shape of the run is what tells a
    // reader whether the JIT was still working or the allocation is real.
    if (taken == MAX_WINDOWS) {
        println("allocation probe did not settle in $taken windows: first=$first last=$last best=$best")
    }
    return best
}

/** How much a window must beat the best seen to count as the JIT still settling rather than as noise. */
private const val IMPROVEMENT = 0.9

/** The longest this waits for a loop to settle, in windows. */
private const val MAX_WINDOWS = 400
