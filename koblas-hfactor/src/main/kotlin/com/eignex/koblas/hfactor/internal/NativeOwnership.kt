package com.eignex.koblas.hfactor.internal

import java.lang.ref.Cleaner
import java.lang.ref.Reference
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** Coordinates explicit close, cleaner release, in-flight calls, and reachability for one native handle. */
internal class NativeOwnership(private val owner: Any, description: String, release: () -> Unit) {
    private val lifecycle = NativeResourceLifecycle(description, release)
    private val cleanup = cleaner.register(owner, lifecycle)

    fun <R> anchoring(body: () -> R): R = lifecycle.withResource {
        try {
            body()
        } finally {
            Reference.reachabilityFence(owner)
        }
    }

    fun close() {
        cleanup.clean()
    }

    private companion object {
        val cleaner: Cleaner = Cleaner.create()
    }
}

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

    fun close() {
        closing.set(true)
        while (activeCalls.get() != 0) Thread.onSpinWait()
        if (released.compareAndSet(false, true)) release()
    }

    override fun run(): Unit = close()
}
