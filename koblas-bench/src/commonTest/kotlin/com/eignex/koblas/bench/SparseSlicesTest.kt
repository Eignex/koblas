package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SparseSlicesTest {
    @Test
    fun `slice cases measure only the selected engine and include validation`() {
        for (operation in listOf("spdot-slice", "spnrm2-slice", "spaxpy-slice", "spscatter-slice")) {
            val case = Cases.parse("$operation+37+sparse-uniform+density=0.5").single()
            for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()) {
                val arm = assertNotNull(sparseArm(case, engine))
                if (engine !== koblas) {
                    assertNull(arm.work)
                    assertContains(assertNotNull(arm.reason), "platform-selected engine")
                } else {
                    val work = assertNotNull(arm.work)
                    assertEquals("default-policy", work.comparisonKind)
                    assertContains(assertNotNull(work.kernel), "portable-index-validation+")
                    val reduction = operation == "spdot-slice" || operation == "spnrm2-slice"
                    assertEquals(if (reduction) "arithmetic" else "reset-and-arithmetic", work.timingMode)
                    val first = work.run()
                    assertTrue(first.isFinite())
                    assertEquals(first, work.run())
                    if (!reduction) assertNotNull(work.result)
                }
            }
        }
    }
}
