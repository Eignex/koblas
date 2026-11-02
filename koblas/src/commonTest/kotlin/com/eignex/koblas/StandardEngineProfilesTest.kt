package com.eignex.koblas

import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.vendor.*
import kotlin.test.*

class StandardEngineProfilesTest {
    @Test
    fun `default selection opens only eligible available providers`() {
        for ((runtime, profile) in listOf(
            "native" to StandardEngineProfiles.nativeLinuxOneMkl,
            "jvm" to StandardEngineProfiles.jvmLinuxOneMkl,
        )) {
            val measured = requireNotNull(profile.target)
            for (host in listOf(measured, HostPlatform(measured.operatingSystem, measured.architecture))) {
                for (access in listOf(false, true)) {
                    for (available in listOf(false, true)) {
                        val opened = mutableListOf<Vendor>()
                        val vendor = provider("onemkl", setOf(Dot, Gemm))
                        val engine = standardEngine(runtime, access, host, owned = null) {
                            opened += it
                            vendor.takeIf { available }
                        }
                        val eligible = host == measured && access

                        assertEquals(if (eligible) listOf(Vendor.OneMkl) else emptyList(), opened)
                        assertEquals(
                            if (eligible && available) vendor else null,
                            engine.wholeCallFor(Gemm, 128, 128, 128, false),
                        )
                        assertEquals("scalar", engine.implementationFor(Sum, 1024))
                    }
                }
            }
        }
    }

    @Test
    fun `an absent host provider retains the supplied owned engine`() {
        val owned = provider("owned", setOf(Sum))

        val engine = configuredEngine(EngineProfile("portable", emptyList()), owned) { error("Unexpected vendor open") }

        assertSame(owned, engine)
    }

    @Test
    fun `source typos fail while unavailable built in providers retain coverage`() {
        for (source in listOf("onemlk", "jvm-smid")) {
            val profile = EngineProfile("typo", listOf(EngineRule(source, setOf(Dot))))

            assertFailsWith<IllegalArgumentException> { configuredEngine(profile, owned = null) { null } }
        }
        val profile = EngineProfile("optional", listOf(EngineRule("onemkl", setOf(Dot))))

        assertSame(BuiltinEngines.scalar, configuredEngine(profile, owned = null) { null })
    }

    private fun provider(id: String, operations: Set<EngineImplementation>): KoblasEngine = object : KoblasEngine {
        override val name = id
        override val implementations = operations
    }
}
