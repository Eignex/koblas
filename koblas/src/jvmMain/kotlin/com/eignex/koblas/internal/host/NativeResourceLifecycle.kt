package com.eignex.koblas.internal.host

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** Serializes explicit or cleaner-driven release against calls using one native resource. */
internal class NativeResourceLifecycle(private val description: String, private val release: () -> Unit) : Runnable {
    private val activeCalls = AtomicInteger()
    private val closing = AtomicBoolean()
    private val released = AtomicBoolean()

    fun <T> withResource(block: () -> T): T {
        while (true) {
            check(!closing.get()) { "$description is closed" }
            val active = activeCalls.get()
            if (!activeCalls.compareAndSet(active, active + 1)) continue
            if (closing.get()) {
                activeCalls.decrementAndGet()
                error("$description is closed")
            }
            try {
                return block()
            } finally {
                activeCalls.decrementAndGet()
            }
        }
    }

    override fun run() {
        closing.set(true)
        while (activeCalls.get() != 0) Thread.onSpinWait()
        if (released.compareAndSet(false, true)) release()
    }
}
