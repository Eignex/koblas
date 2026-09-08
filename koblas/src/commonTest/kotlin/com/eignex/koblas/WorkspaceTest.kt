package com.eignex.koblas

import kotlin.test.*

class WorkspaceTest {

    private fun assertAllDistinct(buffers: List<DoubleArray>, context: String) {
        for (i in buffers.indices) {
            for (j in i + 1 until buffers.size) {
                assertTrue(buffers[i] !== buffers[j], "$context: borrow $i and $j share a buffer")
            }
        }
    }

    @Test
    fun `concurrent borrows are distinct and a released buffer comes back`() {
        val ws = Workspace()
        val first = ws.take(8)
        val second = ws.take(8)
        assertEquals(8, first.size)
        assertEquals(8, second.size)
        assertTrue(first !== second, "two outstanding borrows shared a buffer")
        ws.release(first)
        assertSame(first, ws.take(8), "a released buffer should be handed out again")
    }

    @Test
    fun `pools are keyed by width`() {
        val ws = Workspace()
        val short = ws.take(3)
        val long = ws.take(64)
        assertEquals(3, short.size)
        assertEquals(64, long.size)
        ws.release(short)
        assertSame(short, ws.take(3))
        assertEquals(64, ws.take(64).size)
    }

    @Test
    fun `a pool deepens past its initial capacity`() {
        val ws = Workspace()
        val borrowed = List(9) { ws.take(5) }
        assertAllDistinct(borrowed, "deep pool")
        for (b in borrowed) assertEquals(5, b.size)
        for (b in borrowed) ws.release(b)
        val second = List(9) { ws.take(5) }
        assertAllDistinct(second, "deep pool after release")
    }

    @Test
    fun `the width table grows past its initial capacity`() {
        val ws = Workspace()
        val widths = intArrayOf(1, 2, 4, 7, 16, 33)
        val borrowed = widths.map { ws.take(it) }
        for ((i, w) in widths.withIndex()) assertEquals(w, borrowed[i].size, "width $w")
        assertAllDistinct(borrowed, "many widths")
    }

    @Test
    fun `reserve pre-allocates without lending anything out`() {
        val ws = Workspace()
        ws.reserve(6, count = 3)
        val borrowed = List(3) { ws.take(6) }
        assertAllDistinct(borrowed, "reserved")
        for (b in borrowed) assertEquals(6, b.size)
        ws.reserve(6, count = 1)
        for (b in borrowed) ws.release(b)
        assertAllDistinct(List(3) { ws.take(6) }, "after re-reserve")
    }

    @Test
    fun `integer scratch is reserved and pooled independently`() {
        val ws = Workspace()
        val f64 = ScratchRequirement(size = 6, count = 2)
        ws.reserve(f64)
        ws.reserveI32(6, count = 1)

        assertEquals(2, ws.available(f64))
        val integers = ws.takeI32(6)
        assertEquals(2, ws.available(f64))
        ws.release(integers)
        assertSame(integers, ws.takeI32(6))
    }

    @Test
    fun `integer borrow returns reserved scratch after failure`() {
        val ws = Workspace()
        val parked = ws.takeI32(4)
        ws.release(parked)
        assertFailsWith<IllegalStateException> { ws.borrowI32(4) { error("boom") } }
        assertSame(parked, ws.takeI32(4))
    }

    @Test
    fun `borrow returns the buffer even when the block throws`() {
        val ws = Workspace()
        val seen = ws.borrow(4) { it }
        assertSame(seen, ws.take(4), "borrow did not return its buffer")
        ws.release(seen)
        assertFailsWith<IllegalStateException> { ws.borrow(4) { error("boom") } }
        assertSame(seen, ws.take(4), "a throwing block must still return the buffer")
    }

    @Test
    fun `zero-length borrows are handed out like any other width`() {
        val ws = Workspace()
        val empty = ws.take(0)
        assertEquals(0, empty.size)
        ws.release(empty)
        assertSame(empty, ws.take(0))
    }

    @Test
    fun `releasing a foreign buffer is an error`() {
        val ws = Workspace()
        assertFailsWith<IllegalStateException> { ws.release(DoubleArray(4)) }
        ws.release(ws.take(4))
        assertFailsWith<IllegalStateException> { ws.release(DoubleArray(4)) }
    }

    @Test
    fun `releasing a buffer twice is an error`() {
        val ws = Workspace()
        val first = ws.take(4)
        ws.release(first)

        assertFailsWith<IllegalStateException> { ws.release(first) }
    }

    @Test
    fun `a double release leaves the remaining borrows distinct`() {
        val ws = Workspace()
        val held = ws.take(4)
        val returned = ws.take(4)
        ws.release(returned)

        assertFailsWith<IllegalStateException> { ws.release(returned) }
        assertAllDistinct(listOf(held, ws.take(4), ws.take(4)), "after a rejected double release")
    }

    @Test
    fun `negative sizes and counts are rejected`() {
        val ws = Workspace()
        assertFailsWith<IllegalArgumentException> { ws.take(-1) }
        assertFailsWith<IllegalArgumentException> { ws.reserve(4, count = -1) }
    }

    @Test
    fun `changing sizes does not retain every buffer`() {
        val ws = Workspace()
        for (width in 1..300) {
            ws.release(ws.take(width))
            ws.release(ws.takeI32(width))
        }

        assertTrue(ws.pooledWidths <= 64, "300 double sizes left ${ws.pooledWidths} sizes retained")
        assertTrue(ws.pooledI32Widths <= 64, "300 index sizes left ${ws.pooledI32Widths} sizes retained")
    }

    @Test
    fun `the retained size cap survives a burst of active borrows`() {
        val ws = Workspace()
        val held = (1..64).map { ws.take(it) }
        ws.release(ws.take(65))
        held.forEach { ws.release(it) }
        for (width in 66..140) ws.release(ws.take(width))

        assertTrue(ws.pooledWidths <= 64, "the burst left ${ws.pooledWidths} sizes retained")
    }

    /** Buffers lent out are never reclaimed, however many other widths pass through afterwards. */
    @Test
    fun `an outstanding borrow survives churn through other widths`() {
        val ws = Workspace()
        val held = ws.take(9)
        held[0] = 5.0
        for (width in 100..300) ws.release(ws.take(width))
        assertEquals(5.0, held[0], "a held buffer was handed to someone else")
        ws.release(held)
        assertSame(held, ws.take(9), "the held buffer's pool was dropped while it was lent")
    }

    /** With no workspace to lend one, [borrow] allocates and has nothing to hand back. */
    @Test
    fun `a borrow without a workspace allocates and returns the block value`() {
        val absent: Workspace? = null
        val size = absent.borrow(12) { it.size }
        assertEquals(12, size)
        assertFailsWith<IllegalStateException> { absent.borrow(4) { error("propagates") } }
    }
}
