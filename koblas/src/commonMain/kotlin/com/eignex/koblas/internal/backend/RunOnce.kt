package com.eignex.koblas.internal.backend

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/**
 * Runs an action at most once for the life of the process, and lets a second caller see all of it.
 *
 * The first caller runs the action. A caller on another thread waits, on a platform that has something to
 * wait on, so what it sees afterwards is everything the pass did rather than however far the pass had got.
 * Returning as soon as the pass had started, which is the cheaper thing to do, hands that caller a result
 * still being assembled. Which platforms those are is [withRunOnceLock]'s to say.
 *
 * A caller arriving from inside the action is a different matter, and is what a discovery pass gets when it
 * probes a provider that asks for discovery in turn. That caller cannot wait for the pass it is already
 * inside, and does not: it holds the lock already, and the gate sends it straight back out.
 *
 * A pass that throws leaves the gate open. What it registered before the throw stands, and the next caller
 * runs it again, rather than one failure settling the question for the process.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class RunOnce {
    private val started = AtomicInt(0)

    /** Runs [action] unless it has run, or is running on this thread. Rethrows whatever it raises. */
    @Suppress("TooGenericExceptionCaught") // whatever a pass raises, the gate has to reopen before it leaves
    fun run(action: () -> Unit): Unit = withRunOnceLock {
        if (!started.compareAndSet(0, 1)) return@withRunOnceLock
        try {
            action()
        } catch (raised: Throwable) {
            started.store(0)
            throw raised
        }
    }
}

/**
 * Runs [block] holding whatever this platform has for a lock one thread may re-enter, so a second thread
 * waits for the first. It has to be one a thread may re-enter, since the caller may be the pass itself.
 *
 * A platform with nothing of the sort runs [block] as it is, and each actual says what that leaves open.
 */
internal expect fun <T> withRunOnceLock(block: () -> T): T
