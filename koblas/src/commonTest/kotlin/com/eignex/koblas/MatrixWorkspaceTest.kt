package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * What a reusable workspace keeps and what it lets go.
 *
 * Two behaviours pull against each other. A repeated call over one shape has to get its buffers back, or the
 * workspace saves nothing; a caller sweeping changing shapes must not accumulate every buffer it has ever
 * asked for, or the workspace grows with the program's history. These check both ends and the nesting that
 * the sparse scheduling depends on.
 */
class MatrixWorkspaceTest {

    @Test
    fun `a repeated loan of one length returns the same buffer`() {
        val workspace = MatrixWorkspace()

        val first = workspace.borrow(64) { it }
        val second = workspace.borrow(64) { it }

        assertSame(first, second, "a returned buffer was not lent again")
        assertEquals(1, workspace.available(64))
    }

    @Test
    fun `nested loans of one length are distinct buffers`() {
        val workspace = MatrixWorkspace()

        workspace.borrow(32) { outer ->
            workspace.borrow(32) { inner ->
                assertTrue(outer !== inner, "one buffer was lent twice at the same time")
                inner[0] = 1.0
            }
        }

        assertEquals(2, workspace.available(32), "both loans came back")
    }

    @Test
    fun `a loan is returned when the work it was lent for throws`() {
        val workspace = MatrixWorkspace()

        assertFailsWith<IllegalStateException> {
            workspace.borrow(16) { error("the kernel failed") }
        }

        assertEquals(1, workspace.available(16), "a failed call stranded its loan")
    }

    @Test
    fun `index loans behave as floating-point ones do`() {
        val workspace = MatrixWorkspace()

        val first = workspace.borrowI32(8) { it }
        workspace.borrowI32(8) { inner -> assertSame(first, inner) }

        assertEquals(1, workspace.availableI32(8))
    }

    /**
     * The retention bound, seen from the outside: a workspace swept across many lengths keeps a small number
     * of them rather than all of them. Without the bound this count would be the number of distinct lengths
     * the sweep asked for, which is what makes a long-lived workspace grow without limit.
     */
    @Test
    fun `a sweep over changing lengths retains a bounded number of them`() {
        val workspace = MatrixWorkspace()

        for (length in 1..64) workspace.borrow(length) { it[0] = length.toDouble() }
        val floating = workspace.idleLengths()
        for (length in 1..64) workspace.borrowI32(length) { it[0] = length }

        assertTrue(floating in 1..8, "retained $floating floating-point lengths")
        assertTrue(workspace.idleI32Lengths() in 1..8, "retained ${workspace.idleI32Lengths()} index lengths")
    }

    @Test
    fun `a sweep keeps the most recently returned lengths`() {
        val workspace = MatrixWorkspace()

        for (length in 1..64) workspace.borrow(length) { it[0] = length.toDouble() }

        assertEquals(1, workspace.available(64), "the last length asked for was dropped")
        assertEquals(0, workspace.available(1), "the first length asked for was kept")
    }

    /**
     * Alternating between two shapes is the case the bound must not break: both stay resident, so neither
     * call allocates after the first pass.
     */
    @Test
    fun `alternating between two lengths keeps both resident`() {
        val workspace = MatrixWorkspace()
        workspace.borrow(128) { it[0] = 1.0 }
        workspace.borrow(256) { it[0] = 1.0 }

        repeat(8) {
            workspace.borrow(128) { buffer -> assertEquals(128, buffer.size) }
            workspace.borrow(256) { buffer -> assertEquals(256, buffer.size) }
        }

        assertEquals(1, workspace.available(128))
        assertEquals(1, workspace.available(256))
    }

    @Test
    fun `a negative length is rejected rather than allocated`() {
        val workspace = MatrixWorkspace()

        assertFailsWith<IllegalArgumentException> { workspace.borrow(-1) { it } }
    }
}
