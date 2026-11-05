package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.EngineImplementation.*
import com.eignex.koblas.internal.kernels.EngineKernels
import kotlin.test.*

class EngineTuningTest {
    @Test
    fun `sve tunings match their width and other widths borrow explicit estimates`() {
        for (runtime in EngineRuntime.entries) {
            val measured = standardTuning(
                runtime,
                HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpuClass = CpuClass.Sve, sveVectorBits = 256),
            )
            val estimate = standardTuning(
                runtime,
                HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpuClass = CpuClass.Sve, sveVectorBits = 512),
            )

            assertTrue(measured.name.contains("sve256"))
            assertFalse(measured.name.startsWith("best-effort-"))
            assertTrue(estimate.evidence.startsWith("Unmeasured best effort"))
        }
    }

    @Test
    fun `x86 tunings select the declared instruction class`() {
        for (runtime in EngineRuntime.entries) {
            for (cpuClass in listOf(CpuClass.Avx2, CpuClass.Avx512)) {
                val tuning = standardTuning(
                    runtime,
                    HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = cpuClass),
                )

                assertTrue(tuning.name.contains(cpuClass.name.lowercase()))
                assertFalse(tuning.name.startsWith("best-effort-"))
            }
        }
    }

    @Test
    fun `target tunings route through available permitted platform vendors`() {
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

                        assertEquals(if (access) candidates + extra else emptySet(), opened)
                        val preferences = standardTuning(runtime, host).rules.mapNotNull { rule ->
                            choices.firstOrNull { it.vendorName.lowercase() == rule.source }
                        }
                        val selected = if (access) preferences.firstOrNull { it in available } else null
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
    fun `unknown operating systems retain owned kernels without host lookup`() {
        for (runtime in EngineRuntime.entries) {
            val host = HostPlatform(OperatingSystem.Other, Architecture.X86_64)

            val tuning = standardTuning(runtime, host)

            assertEquals(
                if (runtime == EngineRuntime.Jvm) setOf("jvm-simd") else emptySet(),
                tuning.rules.map { it.source }.toSet(),
            )
        }
    }

    private fun provider(id: String, operations: Set<EngineImplementation>): KoblasEngine = object : EngineKernels {
        override val name = id
        override val implementations = operations
    }
}
