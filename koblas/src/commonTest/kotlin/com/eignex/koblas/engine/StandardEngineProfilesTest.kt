package com.eignex.koblas.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.EngineImplementation.*
import com.eignex.koblas.internal.portable.PortableEngine
import kotlin.test.*

class StandardEngineProfilesTest {
    @Test
    fun `sve profiles match their width and other widths borrow explicit estimates`() {
        for (runtime in EngineRuntime.entries) {
            val measured = StandardEngineProfiles.forTarget(
                runtime,
                HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpuClass = CpuClass.Sve, sveVectorBits = 256),
            )
            val estimate = StandardEngineProfiles.forTarget(
                runtime,
                HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpuClass = CpuClass.Sve, sveVectorBits = 512),
            )

            assertEquals(CpuClass.Sve, measured.target?.cpuClass)
            assertEquals(256, measured.target?.sveVectorBits)
            assertTrue(estimate.evidence.startsWith("Unmeasured best effort"))
        }
    }

    @Test
    fun `x86 profiles select the declared instruction class`() {
        for (runtime in EngineRuntime.entries) {
            for (cpuClass in listOf(CpuClass.Avx2, CpuClass.Avx512)) {
                val profile = StandardEngineProfiles.forTarget(
                    runtime,
                    HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = cpuClass),
                )

                assertEquals(cpuClass, profile.target?.cpuClass)
                assertFalse(profile.name.startsWith("best-effort-"))
            }
        }
    }

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
                        val preferences = StandardEngineProfiles.forTarget(runtime, host).rules.mapNotNull { rule ->
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
    fun `best effort borrows compatible vendors and matching owned architecture within one runtime`() {
        data class Case(val runtime: EngineRuntime, val os: OperatingSystem, val source: String, val boundary: Int)
        val linux = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, cpuClass = CpuClass.Avx512)
        val mac = HostPlatform(OperatingSystem.MacOs, Architecture.Arm64, cpuClass = CpuClass.Arm64)
        val profiles = listOf(
            EngineProfile(
                "linux-jvm",
                listOf(
                    EngineRule("onemkl", setOf(Dot), rows = 64..128),
                    EngineRule("jvm-simd", setOf(Dot), rows = 8..128),
                ),
                EngineRuntime.Jvm,
                linux,
            ),
            EngineProfile(
                "mac-jvm",
                listOf(
                    EngineRule("accelerate", setOf(Dot), rows = 128..256),
                    EngineRule("jvm-simd", setOf(Dot), rows = 2..256),
                ),
                EngineRuntime.Jvm,
                mac,
            ),
            EngineProfile(
                "linux-native",
                listOf(EngineRule("onemkl", setOf(Dot), rows = 4..128)),
                EngineRuntime.Native,
                linux,
            ),
        )
        for (case in listOf(
            Case(EngineRuntime.Jvm, OperatingSystem.MacOs, "accelerate", 128),
            Case(EngineRuntime.Jvm, OperatingSystem.Windows, "onemkl", 64),
            Case(EngineRuntime.Native, OperatingSystem.Windows, "onemkl", 4),
        )) {
            val profile = assertNotNull(
                bestEffortProfile(
                    case.runtime,
                    HostPlatform(case.os, Architecture.X86_64, cpuClass = CpuClass.Avx512),
                    profiles,
                ),
            )
            val engine = KoblasEngine.combine(
                profile,
                provider("onemkl", setOf(Dot)),
                provider("accelerate", setOf(Dot)),
                provider("jvm-simd", setOf(Dot)),
            )

            assertTrue(profile.name.startsWith("best-effort-"))
            assertTrue(profile.evidence.startsWith("Unmeasured best effort"))
            assertFalse("null" in profile.evidence)
            val donor = if (case.runtime == EngineRuntime.Native) "linux-native" else "linux-jvm"
            assertTrue(donor in profile.name && donor in profile.evidence)
            if (case.os == OperatingSystem.MacOs) assertTrue("mac-jvm" in profile.name && "mac-jvm" in profile.evidence)
            assertEquals(case.source, engine.implementationFor(Dot, case.boundary))
            assertEquals(
                if (case.runtime == EngineRuntime.Jvm) "jvm-simd" else PortableEngine.name,
                engine.implementationFor(Dot, case.boundary - 1),
            )
            assertEquals(
                PortableEngine.name,
                engine.implementationFor(Dot, if (case.runtime == EngineRuntime.Jvm) 7 else 3),
            )
        }
        val unknownHost = HostPlatform(OperatingSystem.Other, Architecture.X86_64)
        val unknown = assertNotNull(bestEffortProfile(EngineRuntime.Jvm, unknownHost, profiles))
        assertEquals(setOf("jvm-simd"), unknown.rules.map { it.source }.toSet())
    }

    private fun provider(id: String, operations: Set<EngineImplementation>): KoblasEngine = object : KoblasEngine {
        override val name = id
        override val implementations = operations
    }
}
