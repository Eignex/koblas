package com.eignex.koblas.vendor

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VendorTest {
    @Test
    fun `windows candidates are dlls with no unix paths`() {
        assertEquals(
            listOf("mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.dll"),
            Vendor.OneMkl.resolvedCandidates(null, OperatingSystem.Windows),
        )
        assertEquals(
            listOf("libopenblas.dll", "openblas.dll"),
            Vendor.OpenBlas.resolvedCandidates(null, OperatingSystem.Windows),
        )
        assertEquals(
            listOf(
                "libblis.dll",
                "AOCL-LibBlis-Win-dll.dll",
                "libblis-mt.dll",
                "AOCL-LibBlis-Win-MT-dll.dll",
                "aocl.dll",
            ),
            Vendor.Aocl.resolvedCandidates(null, OperatingSystem.Windows),
        )
        assertEquals(
            listOf("armpl_lp64.dll", "armpl_lp64_mp.dll"),
            Vendor.ArmPl.resolvedCandidates(null, OperatingSystem.Windows),
        )
        assertEquals(emptyList(), Vendor.Accelerate.resolvedCandidates(null, OperatingSystem.Windows))
    }

    @Test
    fun `cpu manufacturer identifiers agree across linux and windows formats`() {
        for (identifier in listOf("GenuineIntel", "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel")) {
            assertEquals(CpuVendor.Intel, cpuVendorFromIdentifier(identifier))
        }
        for (identifier in listOf("AuthenticAMD", "AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD")) {
            assertEquals(CpuVendor.Amd, cpuVendorFromIdentifier(identifier))
        }
        for (identifier in listOf(null, "", "Intel", "AMD", "notGenuineIntel", "ARMv8")) {
            assertEquals(CpuVendor.Unknown, cpuVendorFromIdentifier(identifier))
        }
    }

    @Test
    fun `linux candidates retain installer prefixes and exclude windows dlls`() {
        val candidates = Vendor.OneMkl.resolvedCandidates("/home/test/", OperatingSystem.Linux)

        assertTrue("/home/test/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3" in candidates)
        assertTrue(candidates.all { it.endsWith(".so") || ".so." in it })
        assertTrue(Vendor.OneMkl.resolvedCandidates(null, OperatingSystem.Linux).none { "{home}" in it })
    }

    @Test
    fun `explicit unix vendors preserve their declared candidates`() {
        for (vendor in Vendor.entries) {
            for (system in listOf(OperatingSystem.Linux, OperatingSystem.MacOs, OperatingSystem.Other)) {
                assertEquals(
                    vendor.resolvedCandidates(null, OperatingSystem.Linux),
                    vendor.resolvedCandidates(null, system),
                )
            }
        }
    }

    @Test
    fun `windows loads only absolute path entries in candidate order`() {
        val paths = Vendor.OpenBlas.loadCandidates(
            null,
            OperatingSystem.Windows,
            ".;relative;C:relative;\\rooted;C:\\libs;\"D:/vendor libs\";\\\\server\\share;C:/libs;;",
        )
        assertEquals(
            listOf(
                "C:/libs/libopenblas.dll",
                "D:/vendor libs/libopenblas.dll",
                "//server/share/libopenblas.dll",
                "C:/libs/openblas.dll",
                "D:/vendor libs/openblas.dll",
                "//server/share/openblas.dll",
            ),
            paths,
        )
        assertEquals(emptyList(), Vendor.OpenBlas.loadCandidates(null, OperatingSystem.Windows, null))
    }
}
