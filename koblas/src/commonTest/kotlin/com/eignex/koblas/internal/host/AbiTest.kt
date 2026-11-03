package com.eignex.koblas.internal.host

import com.eignex.koblas.engine.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The rules a library has to satisfy before a caller reaches it, tested against the rule itself, since a host
 * that happens to have one missing exactly the right symbol is not something a test can rely on.
 */
class AbiTest {
    @Test
    fun `a parallel armpl build with hidden controls is never accepted as serial`() {
        for (candidate in listOf("libarmpl_lp64.so", "armpl_lp64.dll", "C:\\parallel_mp\\armpl_lp64.dll")) {
            assertTrue(ThreadControl.OpenMp.permitsAbsentControl(candidate))
        }
        for (candidate in listOf("libarmpl_lp64_mp.so", "armpl_lp64_mp.dll", "C:\\ArmPL\\ARMPL_LP64_MP.DLL")) {
            assertTrue(!ThreadControl.OpenMp.permitsAbsentControl(candidate))
        }
        for (control in ThreadControl.entries.filterNot { it == ThreadControl.OpenMp }) {
            assertTrue(!control.permitsAbsentControl("library.dll"))
        }
    }

    @Test
    fun `missing exports remove only their own capabilities`() {
        for (missing in listOf("cblas_dgemmt", "cblas_dtrsm", "cblas_dnrm2")) {
            val supplied = exportedImplementations { it != missing }

            assertTrue(EngineImplementation.Dot in supplied)
            assertTrue(EngineImplementation.Gemm in supplied)
            assertTrue(supplied.none { it.entryPoint == missing })
        }
    }

    @Test
    fun `a build advertising sixty four bit integers is not the abi koblas binds`() {
        assertTrue(declaresWideIntegers("OpenBLAS 0.3.32 USE64BITINT DYNAMIC_ARCH"))
        assertTrue(declaresWideIntegers("BLIS 0.9.0 ilp64"))
        assertTrue(declaresWideIntegers("something INT64 something"))
        assertTrue(!declaresWideIntegers("OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH Haswell MAX_THREADS=128"))
        assertTrue(!declaresWideIntegers("Intel(R) oneAPI Math Kernel Library Version 2026.1-Product Build"))
    }

    @Test
    fun `the known answer probe is exactly representable`() {
        // The probe compares for equality, which is only legitimate because these values round-trip exactly.
        val dot = AbiProbe.x.indices.sumOf { AbiProbe.x[it] * AbiProbe.y[it] }

        assertEquals(AbiProbe.DOT, dot)
        assertEquals(4, AbiProbe.identity.size)
        assertEquals(4, AbiProbe.operand.size)
    }
}
