package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.vendor.Vendor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EngineIdentityTest {
    @Test
    fun `combined identity attributes only the supplied capabilities to each source`() {
        val source = object : KoblasEngine {
            override val name = "vendor"
            override val vendor = Vendor.OneMkl
            override val libraryPath = "/mkl"
            override val version = "mkl-version"
            override val implementations = setOf(EngineImplementation.Dot)
        }
        val second = object : KoblasEngine {
            override val name = "second"
            override val vendor = Vendor.OpenBlas
            override val libraryPath = "/openblas"
            override val version = "openblas-version"
            override val implementations = setOf(EngineImplementation.Asum)
        }
        val engine = KoblasEngine.combine(EngineProfile("combined", emptyList()), source, second)

        val identity = engineIdentity("native-default", engine)

        assertTrue(identity.startsWith("native-default/combined/"), identity)
        assertTrue("/oneMKL//mkl/mkl-version/threads=" in identity, identity)
        assertTrue("/OpenBLAS//openblas/openblas-version/threads=" in identity, identity)
        assertTrue("/caps[openblas:Asum]" in identity, identity)
        assertEquals(1, Regex("/caps\\[onemkl:").findAll(identity).count())
        assertTrue(identity.contains("/caps[onemkl:Dot]"), identity)
        assertTrue(identity.contains("/caps[portable:"), identity)
    }
}
