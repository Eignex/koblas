package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

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
    fun `negative sizes and counts are rejected`() {
        val ws = Workspace()
        assertFailsWith<IllegalArgumentException> { ws.take(-1) }
        assertFailsWith<IllegalArgumentException> { ws.reserve(4, count = -1) }
    }

    /**
     * Widths that vary must not each cost a pool for the workspace's lifetime. Past the cap the coldest idle
     * one is recycled, and every borrow keeps working throughout.
     */
    @Test
    fun `many distinct widths stay correct and do not retain every pool`() {
        val ws = Workspace()
        for (width in 1..300) {
            val buffer = ws.take(width)
            assertEquals(width, buffer.size, "width $width")
            ws.release(buffer)
        }
        assertTrue(ws.pooledWidths <= 64, "300 widths left ${ws.pooledWidths} pools alive")
        // A width used after the churn still round-trips, and reuse within one width still recycles.
        val hot = ws.take(7)
        ws.release(hot)
        assertSame(hot, ws.take(7), "a just-released buffer should come back")
    }

    /** Buffers lent out are never reclaimed, however many other widths pass through afterwards. */
    @Test
    fun `an outstanding borrow survives churn through other widths`() {
        val ws = Workspace()
        val held = ws.take(9)
        held[0] = 5.0
        for (width in 100..300) ws.release(ws.take(width))
        assertEquals(5.0, held[0], "a held buffer was handed to someone else")
        assertTrue(ws.pooledWidths <= 65, "the lent pool plus the cap, got ${ws.pooledWidths}")
        ws.release(held)
        assertSame(held, ws.take(9), "the held buffer's pool was dropped while it was lent")
    }

    @Test
    fun `releasing a foreign buffer leaves the pool table untouched`() {
        val ws = Workspace()
        ws.release(ws.take(4))
        val widths = ws.pooledWidths
        assertFailsWith<IllegalStateException> { ws.release(DoubleArray(9)) }
        assertEquals(widths, ws.pooledWidths, "a rejected release opened a pool")
        assertFailsWith<IllegalStateException> { ws.release(DoubleArray(4)) }
        assertEquals(widths, ws.pooledWidths, "a rejected release of a pooled width changed the table")
    }

    @Test
    fun `a rejected release at the width cap keeps the pools that exist`() {
        val ws = Workspace()
        for (width in 1..64) ws.release(ws.take(width))
        val widths = ws.pooledWidths
        val hot = ws.take(64)
        ws.release(hot)
        assertFailsWith<IllegalStateException> { ws.release(DoubleArray(1000)) }
        assertEquals(widths, ws.pooledWidths)
        assertSame(hot, ws.take(64), "the rejected release evicted a live pool")
    }
}
