package com.eignex.koblas.internal.backend

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.native.concurrent.ThreadLocal

/** Whether this thread is already inside the pass, which is what makes the lock below re-entrant. */
@ThreadLocal
private object RunOncePass {
    var inside: Boolean = false
}

/** Free, or held by whichever thread is running the pass. */
@OptIn(ExperimentalAtomicApi::class)
private val holder = AtomicInt(0)

/**
 * A re-entrant spin lock, which is what these targets have to build one from.
 *
 * The standard library offers them nothing a thread may re-enter, and re-entrancy is not optional: a
 * discovery pass that probes a provider asking for discovery in turn arrives back here on its own thread,
 * and a plain lock would hang exactly the case [RunOnce] exists to serve. A thread-local flag separates
 * that caller, which goes straight through, from a second thread, which waits.
 *
 * Running the block unguarded instead would return that second thread into a registry still filling up:
 * [RunOnce] flips its gate before the action runs, so the loser of the race would read whichever halves
 * happened to be registered so far and run its whole workload against them. The pass is short and runs
 * once, so spinning costs less than the machinery to park.
 */
@OptIn(ExperimentalAtomicApi::class)
internal actual fun <T> withRunOnceLock(block: () -> T): T {
    if (RunOncePass.inside) return block()
    while (!holder.compareAndSet(0, 1)) {
        // The holder is running the pass. Nothing to wait on here but the flag it clears on the way out.
    }
    RunOncePass.inside = true
    try {
        return block()
    } finally {
        RunOncePass.inside = false
        holder.store(0)
    }
}
