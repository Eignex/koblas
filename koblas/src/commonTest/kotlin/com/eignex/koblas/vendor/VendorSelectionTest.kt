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
        val host = HostPlatform(OperatingSystem.Other, Architecture.Other, CpuVendor.Unknown)

        assertEquals(emptyList(), Vendor.select(host))
    }

    @Test
    fun `openblas is last wherever it is offered at all`() {
        val linux = OperatingSystem.entries.filter { it != OperatingSystem.MacOs }.flatMap { os ->
            listOf(Architecture.X86_64, Architecture.Arm64).flatMap { architecture ->
                CpuVendor.entries.map { HostPlatform(os, architecture, it) }
            }
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
    fun `macos needs no fallback because accelerate cannot be missing`() {
        val host = HostPlatform(OperatingSystem.MacOs, Architecture.Arm64, CpuVendor.Unknown)

        assertEquals(listOf(Vendor.Accelerate), Vendor.select(host))
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
