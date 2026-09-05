package com.eignex.koblas.internal.host

import kotlin.native.concurrent.ThreadLocal

/**
 * Holds the object a native call is reading, so the cleaner cannot free what it owns underneath the call.
 *
 * The bindings reach their native state as a raw address, and the cleaner that frees it is one of the owning
 * object's own fields, so a receiver nothing else refers to can have that state freed in the middle of its
 * own call. `koblas.factor(a).solve(b)` is that shape.
 *
 * One slot per thread, since a call does not re-enter another on the same thread. Saved and restored rather
 * than cleared, so a nested call cannot strand the outer one; the saved value is itself a stack root for as
 * long as the inner call runs.
 */
@ThreadLocal
internal object NativeAnchor {
    var held: Any? = null
}

internal actual fun <T> keepingReachable(owner: Any, block: () -> T): T {
    val previous = NativeAnchor.held
    NativeAnchor.held = owner
    return try {
        block()
    } finally {
        NativeAnchor.held = previous
    }
}
