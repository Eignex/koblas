package com.eignex.koblas

import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.vendor.*
import kotlin.test.*

class StandardEngineProfilesTest {
    @Test
    fun `target profiles route through available permitted platform vendors`() {
        for ((os, arch, candidates) in listOf(
            Triple(OperatingSystem.MacOs, Architecture.Arm64, setOf(Vendor.Accelerate)),
            Triple(OperatingSystem.MacOs, Architecture.X86_64, setOf(Vendor.Accelerate)),
            Triple(OperatingSystem.Linux, Architecture.X86_64, setOf(Vendor.OneMkl, Vendor.OpenBlas)),
            Triple(OperatingSystem.Linux, Architecture.Arm64, setOf(Vendor.OpenBlas)),
            Triple(OperatingSystem.Windows, Architecture.X86_64, setOf(Vendor.OneMkl, Vendor.OpenBlas)),
            Triple(OperatingSystem.Windows, Architecture.Arm64, setOf(Vendor.OpenBlas)),
        )) {
            for (runtime in EngineRuntime.entries) {
                val host = HostPlatform(os, arch)
                val profile = StandardEngineProfiles.forTarget(runtime, host)
                val extra = if (runtime == EngineRuntime.Jvm && os != OperatingSystem.MacOs) {
                    setOf(if (arch == Architecture.X86_64) Vendor.Aocl else Vendor.ArmPl)
                } else {
                    emptySet()
                }
                val choices = candidates + extra
                for (available in listOf(emptySet(), choices) + choices.map { setOf(it) }) {
                    for (access in listOf(false, true)) {
                        val opened = mutableSetOf<Vendor>()
                        val engine = standardEngine(runtime, access, host, owned = null) { requested ->
                            opened += requested
                            provider(requested.vendorName.lowercase(), setOf(Gemm))
                                .takeIf { requested in available }
                        }

                        assertSame(profile, StandardEngineProfiles.forTarget(runtime, host))
                        assertEquals(if (access) candidates + extra else emptySet(), opened)
                        val selected = if (access) choices.firstOrNull { it in available } else null
                        for (borrowed in listOf(false, true)) {
                            assertEquals(
                                selected?.vendorName?.lowercase(),
                                engine.wholeCallFor(Gemm, 256, 256, 256, borrowed)?.id,
                            )
                            assertNull(engine.wholeCallFor(Gemm, 1, 1, 1, borrowed))
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `matrix preferences bound every extent and the borrowed layout`() {
        val backend = provider("matrix", setOf(Gemm))
        val rule = EngineRule(backend.id, backend.implementations, 2..4, 3..5, 6..8, borrowed = false)
        val engine = KoblasEngine.combine(EngineProfile("bounded", listOf(rule)), backend)

        for (dimension in 0..2) {
            val low = intArrayOf(2, 3, 6)
            val high = intArrayOf(4, 5, 8)
            low[dimension]--
            high[dimension]++
            assertNull(engine.wholeCallFor(Gemm, low[0], low[1], low[2], false))
            assertNull(engine.wholeCallFor(Gemm, high[0], high[1], high[2], false))
        }
        assertSame(backend, engine.wholeCallFor(Gemm, 2, 3, 6, false))
        assertSame(backend, engine.wholeCallFor(Gemm, 4, 5, 8, false))
        assertNull(engine.wholeCallFor(Gemm, 2, 3, 6, true))
    }

    @Test
    fun `profiles use only their explicit owned preferences`() {
        val owned = provider("jvm-simd", setOf(Sum))
        for (rules in listOf(emptyList(), listOf(EngineRule("jvm-simd", setOf(Sum), rows = 32..64)))) {
            val engine = configuredEngine(EngineProfile("explicit", rules), owned) { error("Unexpected vendor open") }

            for (length in listOf(16, 32, 64, 128)) {
                assertEquals(
                    if (rules.isNotEmpty() && length in 32..64) owned.name else PortableEngine.name,
                    engine.implementationFor(Sum, length),
                )
            }
        }
    }

    @Test
    fun `source typos fail while unavailable built in providers retain coverage`() {
        for (source in listOf("onemlk", "jvm-smid")) {
            val profile = EngineProfile("typo", listOf(EngineRule(source, setOf(Dot))))

            assertFailsWith<IllegalArgumentException> { configuredEngine(profile, owned = null) { null } }
        }
        val profile = EngineProfile("optional", listOf(EngineRule("onemkl", setOf(Dot))))

        assertEquals(PortableEngine.name, configuredEngine(profile, owned = null) { null }.implementationFor(Dot, 32))
    }

    private fun provider(id: String, operations: Set<EngineImplementation>): KoblasEngine = object : KoblasEngine {
        override val name = id
        override val implementations = operations
    }
}
