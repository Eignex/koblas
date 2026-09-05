package com.eignex.koblas.internal.host

import java.lang.ref.Cleaner
import java.lang.ref.Reference

/**
 * The cleaner every koblas binding registers a native free with, one for the whole process.
 *
 * A cleaner owns a thread, so one per factorization type would be one thread per type for work that is a
 * handful of frees.
 *
 * Register a named class built from the native state, never a lambda that reads a member of the object being
 * registered: an unqualified member read captures that object, which keeps it strongly reachable from here
 * and means the free never runs. Nothing observes that, so it leaks for the life of the process.
 *
 * A registered free also makes every unfenced native read a race. Once the free can run, an accessor that
 * dereferences a native handle needs `Reference.reachabilityFence` around it, or the object can be collected
 * while the call is still in flight.
 */
internal val nativeCleaner: Cleaner = Cleaner.create()

internal actual fun <T> keepingReachable(owner: Any, block: () -> T): T = try {
    block()
} finally {
    Reference.reachabilityFence(owner)
}

internal actual fun registerNativeCleanup(owner: Any, lifecycle: NativeResourceLifecycle): NativeCleanup {
    val cleanable = nativeCleaner.register(owner, lifecycle)
    return NativeCleanup { cleanable.clean() }
}
