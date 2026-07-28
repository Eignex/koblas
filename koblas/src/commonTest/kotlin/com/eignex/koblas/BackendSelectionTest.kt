package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals

class BackendSelectionTest {

    private class Fake(override val name: String, override val priority: Int) : LinearAlgebra by ReferenceLinearAlgebra

    @Test
    fun `registration keeps the highest priority and install overrides everything`() {
        resetRegisteredLinearAlgebra()
        try {
            registerLinearAlgebra(Fake("low", 5))
            assertEquals("low", koblas.name)
            registerLinearAlgebra(Fake("high", 50))
            assertEquals("high", koblas.name)
            registerLinearAlgebra(Fake("mid", 20)) // weaker than the incumbent: ignored
            assertEquals("high", koblas.name)
            installLinearAlgebra(Fake("manual", -1)) // an explicit install beats any priority
            assertEquals("manual", koblas.name)
            installLinearAlgebra(null)
            assertEquals("high", koblas.name)
        } finally {
            installLinearAlgebra(null)
            resetRegisteredLinearAlgebra()
        }
        assertEquals("reference", koblas.name)
    }

    @Test
    fun `koblasInfo reports both seams`() {
        assertEquals("backend=${koblas.name}, primitives=$mathBackend", koblasInfo)
    }
}
