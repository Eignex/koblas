package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation.*
import com.eignex.koblas.internal.kernels.EngineKernels
import kotlin.test.*

class EngineTuningTest {
    @Test
    fun `automatic copy uses vector kernels only on the measured JVM platform`() {
        val vectors = object : EngineKernels {
            override val name = "copy vectors"
            override val implementations = setOf(Copy)
            override fun requiresUnitStride(implementation: EngineImplementation): Boolean = true
            override fun maximumLength(implementation: EngineImplementation): Int = 8192
        }
        for (runtime in EngineRuntime.entries) {
            for (system in OperatingSystem.entries) {
                for (architecture in Architecture.entries) {
                    val engine = BuiltinEngine(
                        "test",
                        KernelSet(vectors = vectors),
                        tuning = EngineTuning(runtime, HostPlatform(system, architecture), null),
                    )
                    val expected = if (runtime == EngineRuntime.Jvm &&
                        system == OperatingSystem.MacOs && architecture == Architecture.Arm64
                    ) vectors else KoblasEngine.scalar

                    assertSame(expected, engine.vectorProvider(Copy, 4096, contiguous = true))
                    assertSame(KoblasEngine.scalar, engine.vectorProvider(Copy, 64, contiguous = true))
                    assertSame(KoblasEngine.scalar, engine.vectorProvider(Copy, 4096, contiguous = false))
                    assertSame(KoblasEngine.scalar, engine.vectorProvider(Copy, 8193, contiguous = true))
                }
            }
        }
    }

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

    private fun provider(vendor: Vendor, operations: Set<EngineImplementation>): EngineKernels =
        object : EngineKernels {
            override val name = vendor.vendorName
            override val vendor = vendor
            override val implementations = operations
        }
}
