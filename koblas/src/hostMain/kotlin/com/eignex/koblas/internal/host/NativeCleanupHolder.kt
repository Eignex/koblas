@file:OptIn(kotlin.experimental.ExperimentalNativeApi::class)

package com.eignex.koblas.internal.host

import kotlin.native.ref.createCleaner

/**
 * Kotlin/Native has no owner-tracking cleaner, so the cleaner hangs off this handle instead. The binding
 * holds the handle, so the two become unreachable together and the release runs at the same point the JVM's
 * would.
 */
private class NativeCleanupHolder(private val lifecycle: NativeResourceLifecycle) : NativeCleanup {
    @Suppress("unused") // the cleaner runs when this holder becomes unreachable, which is the point
    private val cleaner = createCleaner(lifecycle) { it.close() }

    override fun clean(): Unit = lifecycle.close()
}

internal actual fun registerNativeCleanup(owner: Any, lifecycle: NativeResourceLifecycle): NativeCleanup =
    NativeCleanupHolder(lifecycle)
