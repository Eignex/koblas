package com.eignex.koblas.internal.backend

import com.eignex.koblas.BackendRole
import com.eignex.koblas.isAccelerated
import com.eignex.koblas.koblas
import com.eignex.koblas.withCleanBackends
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class PlatformBackendsTest {
    @Test
    fun `automatic discovery leaves dense BLAS built in`() = withCleanBackends {
        val previous = BackendSlot.entries.associate { slot ->
            slot.selectionKeys.property to System.getProperty(slot.selectionKeys.property)
        }
        BackendSlot.entries.forEach { slot -> System.setProperty(slot.selectionKeys.property, "") }

        try {
            registerPlatformBackends()

            assertEquals("reference", koblas.blas.name)
            assertFalse(koblas.isAccelerated(BackendRole.DENSE_BLAS))
        } finally {
            previous.forEach { (property, value) ->
                if (value == null) System.clearProperty(property) else System.setProperty(property, value)
            }
        }
    }
}
