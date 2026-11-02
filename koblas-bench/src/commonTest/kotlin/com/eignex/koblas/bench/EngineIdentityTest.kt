package com.eignex.koblas.bench

import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.EngineProfile
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.vendor.Vendor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EngineIdentityTest {
    @Test
    fun `combined identity attributes only the supplied capabilities to each source`() {
        val source = object : KoblasEngine() {
            override val name = "vendor"
            override val vendor = Vendor.OneMkl
            override val implementations = setOf(EngineImplementation.Dot)
        }
        val engine = KoblasEngine.combine(EngineProfile("combined", emptyList()), source)

        val identity = engineIdentity("native-default", engine)

        assertEquals(1, Regex("/caps\\[onemkl:").findAll(identity).count())
        assertTrue(identity.contains("/caps[onemkl:Dot]"), identity)
        assertTrue(identity.contains("/caps[portable:"), identity)
    }
}
