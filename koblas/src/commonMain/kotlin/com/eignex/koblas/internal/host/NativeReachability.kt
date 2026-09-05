package com.eignex.koblas.internal.host

/**
 * Runs [block] with [owner] strongly reachable until it returns, which every call dereferencing a native
 * handle needs once a cleaner can free that handle.
 *
 * The JVM has `Reference.reachabilityFence` for this. Kotlin/Native has no equivalent and pinning is no
 * substitute, since its cleaner fires on the owner's own unreachability rather than the resource's, so the
 * native side holds the owner in a global for the duration instead. Global variables are GC roots, which is
 * what makes that a fence.
 */
internal expect fun <T> keepingReachable(owner: Any, block: () -> T): T
