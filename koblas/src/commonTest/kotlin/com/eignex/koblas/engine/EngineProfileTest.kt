package com.eignex.koblas.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.EngineImplementation.*
import kotlin.test.*

class EngineProfileTest {
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
    fun `work boundaries avoid overflow at extreme extents`() {
        val rule = EngineRule("test", setOf(Gemm), work = Long.MAX_VALUE)

        assertTrue(rule.matches(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE, true, false))
        assertFalse(rule.matches(Int.MAX_VALUE, 1, Int.MAX_VALUE, true, false))
        assertFalse(rule.matches(0, Int.MAX_VALUE, Int.MAX_VALUE, true, false))
    }

    @Test
    fun `caller collections cannot change a compiled engine`() {
        val operations = mutableSetOf(Sum)
        val rules = mutableListOf(EngineRule("test", operations))
        val profile = EngineProfile("test", rules)
        val engine = KoblasEngine.combine(profile, provider("test", setOf(Sum)))
        operations.clear()
        rules.clear()

        assertEquals("test", engine.implementationFor(Sum, 8))
        assertEquals(1, profile.rules.size)
    }

    @Test
    fun `profiles require their measured instruction class`() {
        val profile = EngineProfile(
            "measured",
            emptyList(),
            EngineRuntime.Native,
            HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = CpuClass.Avx512),
        )

        for (actual in listOf(null, CpuClass.X86_64, CpuClass.Avx2, CpuClass.Avx512)) {
            val host = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = actual)

            assertEquals(actual == CpuClass.Avx512, profile.appliesTo(EngineRuntime.Native, host))
            assertFalse(profile.appliesTo(EngineRuntime.Jvm, host))
        }
    }

    @Test
    fun `SVE profiles require and match their measured vector width`() {
        val profile = EngineProfile(
            "sve",
            emptyList(),
            EngineRuntime.Jvm,
            HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpuClass = CpuClass.Sve, sveVectorBits = 256),
        )

        for (bits in listOf(null, 128, 256, 512)) {
            val host = HostPlatform(
                OperatingSystem.Linux,
                Architecture.Arm64,
                cpuClass = CpuClass.Sve,
                sveVectorBits = bits,
            )
            assertEquals(bits == 256, profile.appliesTo(EngineRuntime.Jvm, host))
        }
        assertFailsWith<IllegalArgumentException> {
            EngineProfile(
                "unmeasured",
                emptyList(),
                target = HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpuClass = CpuClass.Sve),
            )
        }
    }

    private fun provider(id: String, operations: Set<EngineImplementation>): KoblasEngine = object : KoblasEngine {
        override val name = id
        override val implementations = operations
    }
}
