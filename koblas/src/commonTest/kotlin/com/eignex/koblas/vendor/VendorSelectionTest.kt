package com.eignex.koblas.vendor

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VendorSelectionTest {
    @Test
    fun `macos prefers accelerate on every architecture`() {
        for (architecture in Architecture.entries) {
            val host = HostPlatform(OperatingSystem.MacOs, architecture, CpuVendor.Unknown)

            assertEquals(listOf(Vendor.Accelerate), Vendor.select(host))
        }
    }

    @Test
    fun `linux arm takes armpl whatever the cpu manufacturer is`() {
        for (cpu in CpuVendor.entries) {
            val host = HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpu)

            assertEquals(listOf(Vendor.ArmPl), Vendor.select(host))
        }
    }

    @Test
    fun `x86 order puts the matching manufacturer first`() {
        val amd = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Amd)
        val intel = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Intel)

        assertEquals(listOf(Vendor.Aocl, Vendor.OneMkl), Vendor.select(amd))
        assertEquals(listOf(Vendor.OneMkl, Vendor.Aocl), Vendor.select(intel))
    }

    @Test
    fun `an unidentified x86 manufacturer still has a fallback order`() {
        val host = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Unknown)

        assertEquals(listOf(Vendor.OneMkl, Vendor.Aocl), Vendor.select(host))
    }

    @Test
    fun `an unsupported host selects nothing rather than guessing`() {
        val host = HostPlatform(OperatingSystem.Other, Architecture.Other, CpuVendor.Unknown)

        assertEquals(emptyList(), Vendor.select(host))
    }

    @Test
    fun `openblas is never selected in production`() {
        val hosts = OperatingSystem.entries.flatMap { os ->
            Architecture.entries.flatMap { architecture ->
                CpuVendor.entries.map { HostPlatform(os, architecture, it) }
            }
        }

        for (host in hosts) assertTrue(Vendor.OpenBlas !in Vendor.select(host), "selected on $host")
    }

    @Test
    fun `every selectable vendor is reachable from some host`() {
        val selectable = Vendor.entries.filter { it.selectable }
        val reached = OperatingSystem.entries.flatMap { os ->
            Architecture.entries.flatMap { architecture ->
                CpuVendor.entries.flatMap { Vendor.select(HostPlatform(os, architecture, it)) }
            }
        }.toSet()

        assertEquals(selectable.toSet(), reached)
    }
}
