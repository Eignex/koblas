@file:OptIn(ExperimentalAtomicApi::class)

package com.eignex.koblas.internal.host

import platform.posix.sched_yield
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi

internal actual class NativeResourceLifecycle actual constructor(
    private val description: String,
    private val release: () -> Unit,
) {
    private val activeCalls = AtomicInt(0)
    private val closing = AtomicInt(0)
    private val released = AtomicInt(0)

    actual fun <T> withResource(block: () -> T): T {
        while (true) {
            check(closing.load() == 0) { "$description is closed" }
            val active = activeCalls.load()
            if (!activeCalls.compareAndSet(active, active + 1)) continue
            if (closing.load() != 0) {
                decrementActiveCalls()
                error("$description is closed")
            }
            try {
                return block()
            } finally {
                decrementActiveCalls()
            }
        }
    }

    actual fun close() {
        closing.store(1)
        while (activeCalls.load() != 0) sched_yield()
        if (released.compareAndSet(0, 1)) release()
    }

    private fun decrementActiveCalls() {
        while (true) {
            val active = activeCalls.load()
            check(active > 0) { "native resource call count underflow" }
            if (activeCalls.compareAndSet(active, active - 1)) return
        }
    }
}
