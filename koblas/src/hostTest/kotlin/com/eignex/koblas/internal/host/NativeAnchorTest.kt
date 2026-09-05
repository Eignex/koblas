// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.internal.host

import kotlin.test.*

/**
 * The fence every native binding takes around its calls. Kotlin/Native has no reachability fence of its own,
 * so this one is a global slot, and a slot that is not put back is a leak of everything a thread ever called.
 */
class NativeAnchorTest {
    @Test
    fun `a fenced call holds its owner and lets go afterwards`() {
        val owner = Any()

        val held = keepingReachable(owner) { NativeAnchor.held }

        assertSame(owner, held, "the owner was not anchored for the length of the call")
        assertNull(NativeAnchor.held, "the anchor was left set")
    }

    @Test
    fun `a nested call restores the anchor of the one around it`() {
        val outer = Any()
        val inner = Any()

        val seen = keepingReachable(outer) {
            keepingReachable(inner) { NativeAnchor.held }
            NativeAnchor.held
        }

        assertSame(outer, seen, "the inner call stranded the outer one")
        assertNull(NativeAnchor.held, "the anchor was left set")
    }

    @Test
    fun `a call that throws still lets go`() {
        assertFailsWith<IllegalStateException> { keepingReachable(Any()) { error("native failure") } }

        assertNull(NativeAnchor.held, "a failed call left its owner anchored")
    }
}
