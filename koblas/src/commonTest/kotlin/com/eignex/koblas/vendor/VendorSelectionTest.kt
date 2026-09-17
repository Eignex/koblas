package com.eignex.koblas.vendor

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VendorSelectionTest {
    @Test
    fun `macos prefers accelerate on every architecture and needs no fallback behind it`() {
        for (architecture in Architecture.entries) {
            val host = HostPlatform(OperatingSystem.MacOs, architecture, CpuVendor.Unknown)

            // A list of one is the whole macOS policy: Accelerate is part of the system and cannot be missing,
            // so there is nothing for a last resort to catch.
            assertEquals(listOf(Vendor.Accelerate), Vendor.select(host))
        }
    }

    @Test
    fun `linux arm takes armpl whatever the cpu manufacturer is`() {
        for (cpu in CpuVendor.entries) {
            val host = HostPlatform(OperatingSystem.Linux, Architecture.Arm64, cpu)

            assertEquals(listOf(Vendor.ArmPl, Vendor.OpenBlas), Vendor.select(host))
        }
    }

    @Test
    fun `x86 order puts the matching manufacturer first`() {
        val amd = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Amd)
        val intel = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Intel)

        assertEquals(listOf(Vendor.Aocl, Vendor.OneMkl, Vendor.OpenBlas), Vendor.select(amd))
        assertEquals(listOf(Vendor.OneMkl, Vendor.Aocl, Vendor.OpenBlas), Vendor.select(intel))
    }

    @Test
    fun `an unidentified x86 manufacturer still has a fallback order`() {
        val host = HostPlatform(OperatingSystem.Linux, Architecture.X86_64, CpuVendor.Unknown)

        assertEquals(listOf(Vendor.OneMkl, Vendor.Aocl, Vendor.OpenBlas), Vendor.select(host))
    }

    @Test
    fun `an unsupported host selects nothing rather than guessing`() {
        // Neither half is enough on its own: the candidates are ELF sonames, so an unsupported operating
        // system rules them out even on a supported architecture, and the reverse.
        val neither = HostPlatform(OperatingSystem.Other, Architecture.Other, CpuVendor.Unknown)
        val architecture = HostPlatform(OperatingSystem.Linux, Architecture.Other, CpuVendor.Unknown)
        val system = HostPlatform(OperatingSystem.Other, Architecture.X86_64, CpuVendor.Intel)

        assertEquals(emptyList(), Vendor.select(neither))
        assertEquals(emptyList(), Vendor.select(architecture))
        assertEquals(emptyList(), Vendor.select(system))
    }

    @Test
    fun `openblas is last wherever it is offered at all`() {
        val linux = listOf(Architecture.X86_64, Architecture.Arm64).flatMap { architecture ->
            CpuVendor.entries.map { HostPlatform(OperatingSystem.Linux, architecture, it) }
        }

        // A tuned library wins wherever one is installed, so the fallback only decides what a host with none
        // does. Anywhere but last would make it the answer on hosts that have something better.
        for (host in linux) {
            val order = Vendor.select(host)
            assertEquals(Vendor.OpenBlas, order.last(), "not the last resort on $host")
            assertTrue(order.size > 1, "nothing is preferred over the fallback on $host")
        }
    }

    @Test
    fun `every vendor is reachable from some host`() {
        val reached = OperatingSystem.entries.flatMap { os ->
            Architecture.entries.flatMap { architecture ->
                CpuVendor.entries.flatMap { Vendor.select(HostPlatform(os, architecture, it)) }
            }
        }.toSet()

        assertEquals(Vendor.entries.toSet(), reached)
    }
}
