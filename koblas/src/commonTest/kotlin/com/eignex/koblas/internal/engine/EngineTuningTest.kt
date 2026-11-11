package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.internal.kernels.KernelProvider
import kotlin.test.*

class EngineTuningTest {
    @Test
    fun `automatic selection stops at the first available library`() {
        for (runtime in EngineRuntime.entries) {
            for (system in OperatingSystem.entries) {
                for (architecture in Architecture.entries) {
                    val platform = HostPlatform(system, architecture, cpuClass = CpuClass.Avx512)
                    val candidates = vendorOrder(runtime, platform)
                    for (available in candidates.indices) {
                        val opened = mutableListOf<Vendor>()

                        val engine = standardEngine(runtime, host = platform, owned = null) { requested ->
                            opened += requested
                            provider(requested, setOf(Gemm)).takeIf { requested == candidates[available] }
                        }

                        assertEquals(candidates.take(available + 1), opened)
                        assertEquals(listOf(candidates[available]), engine.providers.mapNotNull { it.vendor })
                        assertEquals(candidates[available], engine.wholeCallFor(Gemm, 256, 256, 256)?.vendor)
                    }
                }
            }
        }
    }

    @Test
    fun `missing libraries retain owned kernels`() {
        val platform = HostPlatform(OperatingSystem.Linux, Architecture.X86_64)
        val vectors = provider(Vendor.OpenBlas, setOf(Dot))
        val opened = mutableListOf<Vendor>()
        val owned = BuiltinEngine("owned", KernelSet(vectors = vectors))

        val engine = standardEngine(EngineRuntime.Jvm, host = platform, owned = owned) {
            opened += it
            null
        }

        assertEquals(listOf(Vendor.OpenBlas, Vendor.OneMkl, Vendor.Aocl), opened)
        assertSame(vectors, (engine as BuiltinEngine).vectorProvider(Dot, 256, true))
        assertNull(engine.host)
    }

    @Test
    fun `unknown architectures use scalar without probing libraries`() {
        val platform = HostPlatform(OperatingSystem.Linux, Architecture.Other)

        val engine = standardEngine(EngineRuntime.Jvm, host = platform) { error("Unexpected host lookup") }

        assertSame(KoblasEngine.scalar, engine)
    }

    @Test
    fun `native access denied never opens a host library`() {
        val engine = standardEngine(EngineRuntime.Jvm, nativeAccess = false, owned = null) {
            error("Unexpected host lookup")
        }

        assertSame(KoblasEngine.scalar, engine)
    }

    @Test
    fun `missing exports use owned coverage instead of opening a second library`() {
        val platform = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = CpuClass.Avx512)
        val opened = mutableListOf<Vendor>()
        val engine = standardEngine(EngineRuntime.Native, host = platform, owned = null) { requested ->
            opened += requested
            provider(requested, setOf(Dot))
        }

        val product = engine.wholeCallFor(Gemm, 256, 256, 256)
        val shortDot = engine.implementationFor(Dot, 128)
        val longDot = engine.implementationFor(Dot, 256)

        assertEquals(listOf(Vendor.OneMkl), opened)
        assertNull(product)
        assertEquals(KoblasEngine.scalar.name, shortDot)
        assertEquals(Vendor.OneMkl.vendorName, longDot)
    }

    @Test
    fun `host product eligibility bounds work without overflow`() {
        val tuning = EngineTuning(
            EngineRuntime.Native,
            HostPlatform(OperatingSystem.Linux, Architecture.X86_64),
            Vendor.OpenBlas,
        )

        assertFalse(tuning.hostEligible(Gemm, 8, 8, 7, false))
        assertTrue(tuning.hostEligible(Gemm, 8, 8, 8, false))
        assertTrue(tuning.hostEligible(Gemm, Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE, false))
        assertFalse(tuning.hostEligible(Gemm, 8, 8, 0, false))
    }

    @Test
    fun `jvm scalar reductions use the host only inside the measured window`() {
        for (cpu in listOf(CpuClass.Avx2, CpuClass.Avx512)) {
            val platform = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = cpu)
            val host = provider(Vendor.OpenBlas, setOf(Asum, Iamax))
            val engine = standardEngine(EngineRuntime.Jvm, host = platform, owned = null) { requested ->
                host.takeIf { requested == Vendor.OpenBlas }
            } as BuiltinEngine

            for (operation in listOf(Asum, Iamax)) {
                for (length in listOf(4095, 4096, 65536, 65537)) {
                    val expected = if (length in 4096..65536) host else KoblasEngine.scalar

                    assertSame(expected, engine.vectorProvider(operation, length, true))
                    assertSame(KoblasEngine.scalar, engine.vectorProvider(operation, length, false))
                }
            }
        }
    }

    @Test
    fun `jvm SIMD reductions keep their owned provider`() {
        val platform = HostPlatform(OperatingSystem.Linux, Architecture.X86_64)
        val host = provider(Vendor.OpenBlas, setOf(Asum, Iamax))
        val vectors = provider(Vendor.OpenBlas, setOf(Asum, Iamax))
        val owned = BuiltinEngine("owned", KernelSet(vectors = vectors))
        val engine = standardEngine(EngineRuntime.Jvm, host = platform, owned = owned) { host } as BuiltinEngine

        for (operation in listOf(Asum, Iamax)) {
            assertSame(vectors, engine.vectorProvider(operation, 16384, true))
        }
    }

    @Test
    fun `jvm scalar matrix vector calls retain the portable schedule`() {
        val platform = HostPlatform(OperatingSystem.Linux, Architecture.X86_64)
        val operations = setOf(Gemv, GemvTransposed, Symv, Ger, Syr, Syr2, Trmv, TrmvTransposed, Trsv, TrsvTransposed)
        val host = provider(Vendor.OpenBlas, operations)
        val engine = standardEngine(EngineRuntime.Jvm, host = platform, owned = null) { host }

        for (operation in operations) {
            assertNull(engine.wholeCallFor(operation, 1024, 1024))
        }
    }

    @Test
    fun `explicit host selection does not apply automatic length bounds`() {
        val host = provider(Vendor.OpenBlas, setOf(Asum, Iamax))
        val engine = BuiltinEngine("explicit", host = host)

        for (operation in listOf(Asum, Iamax)) {
            for (length in listOf(1, 1048576)) {
                assertSame(host, engine.vectorProvider(operation, length, true))
            }
        }
    }

    private fun provider(vendor: Vendor, operations: Set<EngineImplementation>): KernelProvider =
        object : KernelProvider {
            override val name = vendor.vendorName
            override val vendor = vendor
            override val implementations = operations
        }
}
