package com.eignex.koblas.internal.host

/**
 * Everything a koblas binding owes the native memory it owns, in one object rather than one copy per binding.
 *
 * A binding holds an address the library allocated and registers a release for it, which puts three
 * obligations on every call that dereferences that address: the resource must still be open, it must not be
 * released while the call runs, and the owner must stay reachable so its own cleaner does not fire underneath
 * it. [anchoring] discharges all three, and is what a binding wraps each of its native calls in.
 *
 * @param owner the object whose reachability governs the cleaner, which is the binding itself.
 * @param description names the resource in the failure a call after [close] raises.
 * @param release frees the native state. It must not read [owner]: on the JVM a lambda capturing it would
 *   keep it reachable from the cleaner, and the free would never run.
 */
internal class NativeOwnership(private val owner: Any, description: String, release: () -> Unit) {
    private val lifecycle = NativeResourceLifecycle(description, release)
    private val cleanup = registerNativeCleanup(owner, lifecycle)

    /** Runs [body] with the resource held open and [owner] fenced against its own cleaner. */
    fun <R> anchoring(body: () -> R): R = lifecycle.withResource { keepingReachable(owner, body) }

    /** Releases the resource, waiting for the calls in flight. Idempotent. */
    fun close() {
        cleanup.clean()
    }
}
