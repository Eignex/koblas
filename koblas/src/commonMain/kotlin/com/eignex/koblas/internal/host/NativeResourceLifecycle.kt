package com.eignex.koblas.internal.host

/**
 * The ownership contract every koblas binding over a native resource keeps, stated once for both platforms.
 *
 * A binding owns memory the library allocated, and two things can release it: the caller, through `close`,
 * and the platform's cleaner, once the owning object is unreachable. Either can happen while a call is
 * reading that memory from another thread, so this guard sits between them. [withResource] admits a call
 * only while the resource is open and counts it for as long as it runs; [close] refuses further calls, waits
 * for the ones in flight, and releases exactly once however many times it is called.
 *
 * What this does not do is keep the owner reachable: a receiver nothing else refers to can be collected in
 * the middle of its own call, which the cleaner then answers by freeing what the call is reading.
 * [keepingReachable] is the other half, and a binding that dereferences a native handle needs both.
 *
 * @param description names the resource in the failure a call after close raises.
 * @param release frees the native state, and must not read the owning object: on the JVM a lambda capturing
 *   it would keep it reachable from the cleaner and the free would never run.
 */
internal expect class NativeResourceLifecycle(description: String, release: () -> Unit) {
    /** Runs [block] with the resource held open, or raises if it is already closing. */
    fun <T> withResource(block: () -> T): T

    /** Closes to further calls, waits for those in flight, and releases once. Idempotent. */
    fun close()
}
