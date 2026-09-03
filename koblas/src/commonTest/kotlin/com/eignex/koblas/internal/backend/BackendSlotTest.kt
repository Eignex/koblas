package com.eignex.koblas.internal.backend

import com.eignex.koblas.BackendRole
import com.eignex.koblas.F64ContextBuilder
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The slot table is what both selection paths read, so what is worth pinning is that it covers every role
 * and that its separate answers about one half agree with each other.
 */
class BackendSlotTest {

    @Test
    fun `every role has exactly one half`() {
        assertEquals(BackendRole.entries.size, BackendSlot.entries.size)
        assertEquals(BackendRole.entries.toSet(), BackendSlot.entries.mapTo(mutableSetOf()) { it.role })
        BackendRole.entries.forEach { role -> assertEquals(role, role.slot.role, "$role reads back as itself") }
    }

    /** A half added by copying a neighbouring entry would otherwise share the neighbour's pin. */
    @Test
    fun `every half has selection keys of its own`() {
        val properties = BackendSlot.entries.map { it.selectionKeys.property }
        val environments = BackendSlot.entries.map { it.selectionKeys.environment }

        assertEquals(properties.distinct(), properties)
        assertEquals(environments.distinct(), environments)
    }

    @Test
    fun `every required half accepts its own portable default`() {
        BackendSlot.contextHalves.forEach { slot ->
            val portable = slot.portableDefault()
            assertTrue(slot.accepts(portable), "${slot.name} rejects its default ${portable.name}")
        }
    }

    @Test
    fun `every required half reads back something it accepts`() {
        val context = F64ContextBuilder().resolve()

        BackendSlot.contextHalves.forEach { slot ->
            val selected = slot.from(context)
            assertTrue(slot.accepts(selected), "${slot.name} read back ${selected.name}")
        }
    }

    /** The optional half stands a placeholder in instead, which is what makes its role report as absent. */
    @Test
    fun `the repeated LU half defaults to the placeholder it reports as unavailable`() {
        val placeholder = BackendSlot.F64RepeatedSparseLu.portableDefault()

        assertFalse(BackendSlot.F64RepeatedSparseLu.accepts(placeholder))
        assertFalse(placeholder.isAvailable)
    }
}
