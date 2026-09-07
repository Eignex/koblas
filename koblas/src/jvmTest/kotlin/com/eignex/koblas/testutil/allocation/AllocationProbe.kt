package com.eignex.koblas.testutil.allocation

import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory

/** The JVM's per-thread allocation counter, which is where a byte-level measurement has to come from. */
private val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean

/** Holds each block's result so escape analysis cannot delete the allocation being measured. */
internal var allocationSink: Any? = null

/**
 * Bytes [block] allocates per call, as the smallest of [windows] measurement windows of [iterations] calls.
 * The minimum is what the loop costs once the JIT has settled; the other windows carry runtime noise.
 *
 * [block] deliberately takes no iteration index. A `(Int) -> Any?` would box one on every call, and above
 * `Integer`'s cache that is a sixteen-byte allocation charged to whatever is being measured.
 */
internal fun bytesPerIteration(iterations: Int, warmup: Int = 200, windows: Int = 5, block: () -> Any?): Double {
    repeat(warmup) { allocationSink = block() } // let the JIT settle, since the first calls allocate profiling data
    val id = Thread.currentThread().threadId()
    var best = Double.MAX_VALUE
    repeat(windows) {
        val before = bean.getThreadAllocatedBytes(id)
        repeat(iterations) { allocationSink = block() }
        val after = bean.getThreadAllocatedBytes(id)
        best = minOf(best, (after - before).toDouble() / iterations)
    }
    return best
}
