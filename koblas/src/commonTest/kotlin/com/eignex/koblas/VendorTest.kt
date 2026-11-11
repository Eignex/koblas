package com.eignex.koblas

import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.engine.cpuClass
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
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
    fun `linux candidates retain installer prefixes and exclude windows dlls`() {
        val candidates = Vendor.OneMkl.resolvedCandidates("/home/test/", OperatingSystem.Linux)

        assertTrue("/home/test/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3" in candidates)
        assertTrue(candidates.all { it.endsWith(".so") || ".so." in it })
        assertTrue(Vendor.OneMkl.resolvedCandidates(null, OperatingSystem.Linux).none { "{home}" in it })
    }

    @Test
    fun `explicit unix vendors preserve their declared candidates`() {
        for (vendor in Vendor.entries) {
            for (system in listOf(OperatingSystem.MacOs, OperatingSystem.Other)) {
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

    @Test
    fun `cpu flags name the widest instruction class they report`() {
        assertEquals(CpuClass.Avx512, cpuClass(Architecture.X86_64, "sse2 avx2 avx512f"))
        assertEquals(CpuClass.Avx2, cpuClass(Architecture.X86_64, "sse2 avx2"))
        assertEquals(CpuClass.X86_64, cpuClass(Architecture.X86_64, "sse2"))
        assertEquals(CpuClass.Arm64, cpuClass(Architecture.Arm64, "fp asimd"))
        assertEquals(CpuClass.Sve, cpuClass(Architecture.Arm64, "fp asimd sve"))
        assertNull(cpuClass(Architecture.Arm64, null))
        assertNull(cpuClass(Architecture.X86_64, null))
    }
}
